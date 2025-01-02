// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolov8 torchscript
//      yolo export model=yolov8n.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolov8n.torchscript
// 4. modify yolov8n_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_165 = v_142.view(1, 144, 6400)
//          v_166 = v_153.view(1, 144, 1600)
//          v_167 = v_164.view(1, 144, 400)
//          v_168 = torch.cat((v_165, v_166, v_167), dim=2)
//          ...
//      after:
//          v_165 = v_142.view(1, 144, -1).transpose(1, 2)
//          v_166 = v_153.view(1, 144, -1).transpose(1, 2)
//          v_167 = v_164.view(1, 144, -1).transpose(1, 2)
//          v_168 = torch.cat((v_165, v_166, v_167), dim=1)
//          return v_168
// 5. re-export yolov8 torchscript
//      python3 -c 'import yolov8n_pnnx; yolov8n_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolov8n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
// 7. now you get ncnn model files
//      mv yolov8n_pnnx.py.ncnn.param yolov8n.ncnn.param
//      mv yolov8n_pnnx.py.ncnn.bin yolov8n.ncnn.bin

// the out blob would be a 2-dim tensor with w=144 h=8400
//
//        | bbox-reg 16 x 4       | per-class scores(80) |
//        +-----+-----+-----+-----+----------------------+
//        | dx0 | dy0 | dx1 | dy1 |0.1 0.0 0.0 0.5 ......|
//   all /|     |     |     |     |           .          |
//  boxes |  .. |  .. |  .. |  .. |0.0 0.9 0.0 0.0 ......|
//  (8400)|     |     |     |     |           .          |
//       \|     |     |     |     |           .          |
//        +-----+-----+-----+-----+----------------------+
//

#include "yolov8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4; // number of classes. 80 for COCO

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                score = sigmoid(score);
            }

            if (score >= prob_threshold)
            {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4);

                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(pred_bbox, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);
        pred_row_offset += num_grid;
    }
}

int YOLOv8_det::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;//640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // ultralytics/cfg/models/v8/yolov8.yaml
    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // letterbox pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> proposals;
    generate_proposals(out, strides, in_pad, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int YOLOv8_det_coco::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "insan", "bisiklet", "araba", "motosiklet", "ucak", "otobus", "tren", "kamyon", "tekne", "trafik isigi",
        "yangin muslugu", "dur tabelasi", "park sayaci", "bank", "kus", "kedi", "kopek", "at", "koyun", "inek",
        "fil", "ayi", "zebra", "zurafa", "sirt cantasi", "semsiye", "el cantasi", "kravat", "bavul", "frizbi",
        "kayak", "snowboard", "spor topu", "ucurtma", "beyzbol sopasi", "beyzbol eldiveni", "kaykay", "sorf tahtasi",
        "tenis raketi", "sise", "sarap bardagi", "fincan", "catal", "bicak", "kasik", "kase", "muz", "elma",
        "sandvic", "portakal", "brokoli", "havuc", "sosisli sandvic", "pizza", "donut", "pasta", "sandalye", "kanepe",
        "saksi bitkisi", "yatak", "yemek masasi", "tuvalet", "tv", "laptop", "fare", "uzaktan kumanda", "klavye", "cep telefonu",
        "mikrodalga", "firin", "tost makinesi", "lavabo", "buzdolabi", "kitap", "saat", "vazo", "makas", "oyuncak ayi",
        "sac kurutma makinesi", "dis fircasi"
};

    static cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(rgb, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}

int YOLOv8_det_oiv7::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "Akordion", "Yapiskan bant", "Hava araci", "Ucak", "Alarm saati", "Alpaka", "Ambulans", "Hayvan",
        "Karinca", "Antilop", "Elma", "Armadillo", "Enginar", "Araba parcasi", "Balta", "Sirt cantasi", "Bagel",
        "Firinda pisirilmis urun", "Denge cubugu", "Top", "Balon", "Muz", "Yara bandi", "Banjo", "Sandal", "Fici",
        "Beyzbol sopasi", "Beyzbol eldiveni", "Yarasa", "Banyo aksesuari", "Banyo dolabi", "Kupa",
        "Deney cami", "Ayi", "Yatak", "Ari", "Arikovani", "Bira", "Bocek", "Dolmalik biber", "Kemer", "Bank",
        "Bisiklet", "Bisiklet kaski", "Bisiklet tekerlegi", "Bide", "Reklam panosu", "Bilardo masasi", "Durbun",
        "Kus", "Blender", "Mavi alakarga", "Tekne", "Bomba", "Kitap", "Kitaplik", "Bot", "Sise", "Sise acici",
        "Yay ve ok", "Kase", "Bowling ekipmani", "Kutu", "Erkek cocuk", "Sutyen", "Ekmek", "Evrak cantasi",
        "Brokoli", "Bronz heykel", "Kahverengi ayi", "Bina", "Bog", "Durum", "Otobus", "Bust", "Kelebek",
        "Lahana", "Mobilya dolabi", "Pasta", "Pasta standi", "Hesap makinesi", "Deve", "Kamera", "Kutu acici", "Kanarya",
        "Mum", "Sekerleme", "Top", "Kano", "Kavun", "Araba", "Etcil hayvan", "Havuc", "El arabasi", "Kaset calar",
        "Kale", "Kedi", "Kedi mobilyasi", "Tirtir", "Sigir", "Tavan ventilatoru", "Viyolonsel", "Kirkkuyruk",
        "Zincirli testere", "Sandalye", "Peynir", "Cheetah", "Cekmeceli dolap", "Tavuk", "Cingirak", "Keski", "Cubuk",
        "Yilbasi agaci", "Saat", "Dolap", "Giysi", "Ceket", "Kokteyl", "Kokteyl sarkaci", "Hindistancevizi",
        "Kahve", "Kahve fincani", "Sehpa", "Kahve makinesi", "Para", "Incir", "Gunes cicegi",
        "Bilgisayar klavyesi", "Bilgisayar monitoru", "Bilgisayar faresi", "Kap", "Market", "Kurabiye",
        "Pisirici sprey", "Makarali telefon", "Kozmetik", "Kanepe", "Tezgah", "Kovboy sapkasi", "Yengec", "Krem",
        "Kriket topu", "Timsah", "Kruvasan", "Tac", "Koltuk deynegi", "Salatalik", "Mutfak dolabi", "Perde",
        "Kesme tahtasi", "Hancer", "Sut urunu", "Geyik", "Masa", "Tatli", "Bebek bezi", "Zar", "Dijital saat",
        "Dinozor", "Bulasic makinesi", "Kopek", "Kopek yatagi", "Bebek", "Yunus", "Kapi", "Kapi kolu", "Donut",
        "Yusufcuk", "Cekmece", "Elbise", "Matkap", "Icecek", "Icecek pipeti", "Davul", "Ordek", "Halter",
        "Kartal", "Kupe", "Yumurta", "Fil", "Zarf", "Silgi", "Yuz tozu", "Yuz mendili tutucu",
        "Sahin", "Moda aksesuari", "Fast food", "Faks", "Fedora", "Dosya dolabi", "Yangin hidranti",
        "Somine", "Balik", "Bayrak", "Fener", "Cicek", "Cicek saksisi", "Flut", "Ucan disk", "Yiyecek",
        "Mutfak robotu", "Futbol", "Futbol kaski", "Ayakkabi", "Catal", "Cesme", "Tilki", "Patates kizartmasi",
        "Fransiz kornasi", "Kurbaga", "Meyve", "Kizartma tavasi", "Mobilya", "Asparagus", "Gaz ocagi", "Zurafa",
        "Kiz cocuk", "Gozluk", "Eldiven", "Keci", "Koruyucu gozluk", "Japon baligi", "Golf topu", "Golf arabasi", "Gondol",
        "Kaz", "Uzum", "Greyfurt", "Ogutucu", "Guacamole", "Gitar", "Sac kurutma makinesi", "Sac spreyi", "Hamburger",
        "Cekic", "Hamster", "El kurutucu", "Canta", "Tabanca", "Habor foku", "Harmonica", "Harp",
        "Klavsen", "Sapka", "Kulaklik", "Isitici", "Kirpi", "Helikopter", "Kask", "Yuksek topuklu ayakkabi",
        "Doga yuruyus ekipmani", "Hipo", "Cihaz", "Petek", "Bar", "At", "Sosisli sandvic",
        "Ev", "Ev bitkisi", "Kol", "Sakal", "Vucut", "Kulak", "Goz", "Yuz",
        "Ayak", "Sac", "El", "Bas", "Bacak", "Agiz", "Burun",
        "Nemlendirici", "Dondurma", "Kurek cekicisi", "Besik", "Bocek", "Omurgasiz", "Ipod", "Isopod",
        "Ceket", "Jakuzi", "Jaguar", "Kot pantolon", "Denizanasi", "Jet ski", "Kup", "Meyve suyu", "Kanguru",
        "Caydanlik", "Mutfak ve yemek masasi", "Mutfak cihazi", "Mutfak bicagi", "Mutfak gereci",
        "Mutfak gerecleri", "Ucurtma", "Bicak", "Koala", "Merdiven", "Kepce", "Ugur bocegi", "Lamba", "Kara araci",
        "Fener", "Dizustu bilgisayar", "Lavanta (Bitki)", "Limon", "Leopar", "Ampul", "Lamba dugmesi", "Deniz feneri",
        "Zambak", "Limuzin", "Aslan", "Ruj", "Kertenkele", "Istakoz", "Ikili koltuk", "Bagaj ve cantalar", "Vasak",
        "Saksagan", "Memeli", "Adam", "Mango", "Akcaagac", "Marakas", "Deniz omurgasizlari", "Deniz memelisi",
        "Olcu kabi", "Mekanik fan", "Medikal ekipman", "Mikrofon", "Mikrodalga firin", "Sut",
        "Mini etek", "Ayna", "Fuze", "Karistirici", "Karistirma kasesi", "Cep telefonu", "Maymun", "Kelebekler ve guveler",
        "Motosiklet", "Fare", "Muffin", "Kupa", "Katir", "Mantar", "Muzik aleti", "Muzik klavyesi",
        "Civi", "Kolye", "Komodin", "Obua", "Ofis binasi", "Ofis malzemeleri", "Portakal",
        "Org", "Devekusu", "Su samuru", "Firin", "Baykus", "Istiridye", "Kurek", "Palmiye agaci",
        "Pankek", "Panda", "Kagit kesici", "Kagit havlu", "Parasut", "Park metre", "Papagan", "Makarna",
        "Hamur isi", "Seftali", "Armut", "Kalem", "Kalem kutusu", "Kalemtiras", "Penguen", "Parfum", "Kisi",
        "Kisisel bakim", "Kisisel can yelegi", "Piyano", "Piknik sepeti", "Fotograf cercevesi", "Domuz",
        "Yastik", "Ananas", "Surahi", "Pizza", "Pizza kesici", "Bitki", "Plastik torba", "Tabak",
        "Servis tabagi", "Sihhi tesisat ekipmani", "Kutup ayisi", "Nar", "Patlamis misir", "Veranda", "Kirpi", "Poster",
        "Patates", "Priz ve fis", "Duduklu tencere", "Simit", "Yazici", "Kabak", "Boks torbasi",
        "Tavsan", "Rakun", "Raket", "Turp", "Mandal", "Karga", "Isinlar ve patenler", "Kirmizi panda",
        "Buzdolabi", "Uzaktan kumanda", "Surungen", "Gergedan", "Tufek", "Klasor", "Roket",
        "Paten", "Gul", "Ragbi topu", "Cetvel", "Salata", "Tuz ve biberlik", "Sandal",
        "Sandvic", "Fincan tabagi", "Saksafon", "Terazi", "Atki", "Makas", "Skor tablosu", "Akrep",
        "Tornavida", "Heykel", "Deniz aslani", "Deniz kaplumbagasi", "Deniz urunleri", "Denizati", "Emniyet kemeri", "Segway",
        "Servis tepsisi", "Dikis makinesi", "Kopekbaligi", "Koyun", "Raf", "Kabuklu deniz hayvanlari", "Gomlek", "Sort",
        "Pompali tufek", "Dus", "Karides", "Lavabo", "Kaykay", "Kayak", "Etek", "Kafatasi", "Kokarca", "Gokdelen",
        "Yavas pisirici", "Atistirmalik", "Salyangoz", "Yilan", "Snowboard", "Kardan adam", "Kar motoru", "Kar kureme araci",
        "Sabunluk", "Corap", "Kanepe yatagi", "Sombrero", "Serce", "Spatula", "Baharatlik", "Orumcek",
        "Kasik", "Spor ekipmanlari", "Spor formasi", "Kabak", "Kalamar", "Sincap", "Merdivenler",
        "Zimba", "Deniz yildizi", "Sabit bisiklet", "Steteskop", "Tabure", "Dur isareti", "Cilek",
        "Sokak lambasi", "Sedye", "Studyo kanepe", "Denizalti", "Denizalti sandvici", "Takim elbise", "Bavul",
        "Gunes sapkasi", "Gunes gozlugu", "Sorf tahtasi", "Sushi", "Kugu", "Yuzme bonesi", "Havuz", "Mayo",
        "Kilic", "Siringa", "Masa", "Masa tenisi raketi", "Tablet bilgisayar", "Sofra takimi", "Tako", "Tank",
        "Musluk", "Tart", "Taksi", "Cay", "Caydanlik", "Oyuncak ayi", "Telefon", "Televizyon", "Tenis topu",
        "Tenis raketi", "Cadir", "Tac", "Kene", "Kravat", "Kaplan", "Konserve kutusu", "Lastik", "Tost makinesi", "Tuvalet",
        "Tuvalet kagidi", "Domates", "Alet", "Dis fircasi", "Fener", "Kaplumbaga", "Havlu", "Kule", "Oyuncak",
        "Trafik lambasi", "Trafik isareti", "Tren", "Egzersiz banki", "Kosu bandi", "Agac", "Agac evi",
        "Tripod", "Trombon", "Pantolon", "Kamyon", "Trompet", "Hindi", "Kaplumbaga", "Semsiye", "Tek tekerlekli bisiklet",
        "Minibus", "Vazo", "Sebze", "Arac", "Arac plakasi", "Keman", "Voleybol",
        "Waffle", "Waffle demiri", "Duvar saati", "Gardirop", "Camasir makinesi", "Cop kutusu", "Saat",
        "Su araci", "Karpuz", "Silah", "Balina", "Tekerlek", "Tekerlekli sandalye", "Cirpici", "Beyaz tahta", "Sogut",
        "Pencere", "Pencere panjuru", "Sarap", "Sarap bardagi", "Sarap rafi", "Kis kavunu", "Wok", "Kadin",
        "Odun sobasi", "Agackakan", "Solucan", "Anahtar", "Zebra", "Kabak"
};

    static cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(rgb, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}

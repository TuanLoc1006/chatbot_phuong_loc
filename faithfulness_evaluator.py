from typing import Optional

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer



class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

actual_output = "Chương trình đào tạo ngành Y khoa của Đại học Y Dược Cần Thơ bao gồm các hệ sau:\n- Hệ chính quy:\n- Đào tạo trúng tuyển chính quy\n- Đào tạo theo nhu cầu xã hội (theo đặt hàng)\n- Đào tạo hệ quốc tế tiếng Anh\n- Hệ liên thông:\n- Đào tạo liên thông hệ chính quy\n- Đào tạo liên thông theo nhu cầu xã hội\nBạn có thể tham khảo chi tiết chương trình đào tạo tại các liên kết sau:\n- [Chương trình đào tạo ngành Y khoa - Hệ chính quy](http://www.ctump.edu.vn/DesktopModules/NEWS/DinhKem/9079_Nganh-Y-khoa.pdf)\n- [Chương trình đào tạo ngành Y khoa - Hệ quốc tế tiếng Anh](http://www.ctump.edu.vn/DesktopModules/NEWS/DinhKem/9079_Nganh-Y-khoa.pdf)\n- [Chương trình đào tạo ngành Y khoa - Hệ liên thông](http://www.ctump.edu.vn/DesktopModules/NEWS/DinhKem/9079_Nganh-Y-khoa.pdf)\nVui lòng truy cập các liên kết trên để tìm hiểu thêm về chương trình đào tạo cụ thể của từng hệ."

context = "giới thiệu trường: Đại học Y Dược Cần Thơ xây dựng và phát triển trên cơ sở là Khoa Y - Nha - Dược thuộc Trường Đại học Cần Thơ được hình thành vào tháng 7 năm 1979, tọa lạc tại Khu II Trường Đại học Cần Thơ.Năm 1980-1993 Khoa Y chính thức có cơ sở mới đặt tại Khu III Đại học Cần Thơ, tọa lạc tại đường Lý Tự Trọng, phường An Cư, quận Ninh Kiều, thành phố Cần Thơ, gồm một khu nhà 3 tầng (1.500 m²), và 01 văn phòng (50 m²). Thời gian này khoa chỉ có 02 hội trường 200 chỗ ngồi một văn phòng Khoa và 6 phòng vừa làm văn phòng vừa là phòng thực tập của các bộ môn cơ sở: Giải phẫu, Sinh lý, Vi sinh, Sinh lý bệnh, Mô phôi, Giải phẫu bệnh, Ký sinh trùng, Thư viện.Khi tách ra từ trường Đại học Cần Thơ, số lượng cán bộ chỉ có 217 cán bộ (01 Phó Giáo sư, 5 Tiến sĩ, 27 Thạc sĩ, 01 Chuyên khoa cấp II, 04 giảng viên chính, 163 cán bộ giảng). Trường có 04 khoa, bao gồm: khoa Y, khoa Dược, khoa RHM, khoa YTCC và đào tạo 3 mã ngành: Y đa khoa, Răng hàm mặt, Dược học; chưa có mã ngành đào tạo sau đại học. Quy mô đào tạo gần 3.000 sinh viên.\n\nChương trình đào tạo : \nChương trình đào tạo ngành Y khoa/Y đa khoa, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/301.pdf\nChương trình đào tạo ngành Răng hàm mặt, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/302.pdf\nChương trình đào tạo ngành Y học cổ truyền, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/308.pdf\nChương trình đào tạo ngành Y học dự phòng, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/304.pdf\nChương trình đào tạo ngành Dược học, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/303.pdf\nChương trình đào tạo ngành Điều dưỡng, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/305.pdf\nChương trình đào tạo ngành Hộ sinh, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/309.pdf\nChương trình đào tạo ngành Y tế công cộng, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/306.pdf\nChương trình đào tạo ngành Kỹ thuật xét nghiệm y học, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/307.pdf\nChương trình đào tạo ngành Kỹ thuật hình ảnh y học, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/310.pdf\nChương trình đào tạo ngành Kỹ thuật y sinh, xem tại:https://htql.ctump.edu.vn/ctump/mienhp/ctdt/311.pdf\n\ngiới thiệu trường: đại học y dược cần thơ lớn nhất đồng bằng sông cửu long,... sứ mạng: trường đại học y dược cần thơ có sứ mạng đào tạo nguồn nhân lực y tế chất lượng cao; nghiên cứu khoa học, ứng dụng và chuyển giao công nghệ; bảo vệ, chăm sóc và nâng cao sức khỏe nhân dân. tầm nhìn: đến năm 2025: là một trong 05 trường đại học khoa học sức khỏe hàng đầu việt nam và xếp hạng trong 500 trường đại học hàng đầu đông nam á.đến năm 2030: là một trong 05 trường đại học khoa học sức khỏe hàng đầu việt nam và xếp hạng trong 1000 trường đại học hàng đầu châu á. giá trị cốt lỗi: trách nhiệm - chất lượng - phát triển - hội nhập, triết lý giáo dục: trí tuệ - y đức - sáng tạo\n\nThông tin liên hệ phòng đào tạo đại học:địa chỉ: tầng trệt, khoa y, trường đại học y dược cần thơ;điện thoại: 0292.3831.531;email: daotao@ctump.edu.vn \n\nSố khoa, phòng ban, giảng viên : Trường có 6 khoa đào tạo, 56 bộ môn, liên bộ môn trực thuộc khoa; 10 Phòng, ban, 4 Trung tâm, 1 đơn vị và 1 Bệnh viện trực thuộc Trường với 477 giảng viên cơ hữu (trong đó có 2 Giáo sư và 25 Phó Giáo sư; 85 Tiến sĩ, 242 Thạc sĩ; 63 Chuyên khoa cấp II, 4 Chuyên khoa cấp I, và 83 đại học). Đây đều là các bác sĩ, cán bộ khoa học có trình độ cao và có nhiều kinh nghiệm nhiều năm làm việc trong lĩnh vực Y khoa.\n\nCông tác đào tạo: Từ năm 1979, Khoa Y đào tạo khoá bác sĩ y khoa hệ chính quy đầu tiên gồm 95 sinh viên. Đến tháng 9 năm 1983 đào tạo thêm hệ liên thông (chuyên tu) Y đa khoa. Từ đó mỗi năm chỉ đào tạo gần 100 sinh viên y hệ chính quy; từ năm 1984 hàng năm đào tạo 100 đến 150 Bác sĩ y khoa hệ chuyên tu. Năm 1985, khoá đào tạo bác sĩ đa khoa đầu tiên có 95 sinh viên tốt nghiệp và Trường tuyển chọn 20 bác sĩ bồi dưỡng thành cán bộ giảng của khoa Y. Từ đó trở đi mỗi năm, Trường tuyển chọn sinh viên tốt nghiệp loại giỏi, khá tiếp tục đào tạo, bồi dưỡng thành cán bộ giảng dạy của Trường.\n\nNgành Bác sĩ Y khoa:\nĐào tạo chính quy: 924,300 VNĐ/tín chỉ\nĐào tạo theo nhu cầu xã hội (đặt hàng): 1,363,700 VNĐ/tín chỉ\nĐào tạo hệ cử tuyển: 835,900 VNĐ/tín chỉ\nĐào tạo sinh viên quốc tế (chương trình tiếng Việt, ngoài hiệp định): 1,363,700 VNĐ/tín chỉ\nĐào tạo liên thông hệ chính quy: 921,700 VNĐ/tín chỉ\nĐào tạo liên thông theo nhu cầu xã hội (đặt hàng): 1,405,300 VNĐ/tín chỉ\nLưu ý: Mức học phí trên có thể thay đổi theo từng năm học và tùy thuộc vào hình thức đào tạo.\n\nThông tin chỉ tiêu tuyển sinh ngành Y khoa: - Chỉ tiêu kết quả THPT: 99999; - Chỉ tiêu dự bị dân tộc: 25; - Chỉ tiêu hợp đồng: 300;Mã xét tuyển:100;Phương thức xét tuyển:Kết quả tốt nghiệp THPT;Tổ hợp:Toán,Hoá,Sinh\n\nThông tin chỉ tiêu tuyển sinh ngành Y khoa: - Chỉ tiêu kết quả THPT: 710; - Chỉ tiêu dự bị dân tộc: 25; - Chỉ tiêu hợp đồng: 300;Mã xét tuyển:100;Phương thức xét tuyển:Kết quả tốt nghiệp THPT;Tổ hợp:Toán,Hoá,Sinh\n            \nAI: Có thể tham khảo phản hồi này: <br>Chương trình đào tạo của Y khoa : <br>Hệ chính quy :<br>- Đào tạo trúng tuyển chính quy : CTDT Y khoa: http://www.ctump.edu.vn/DesktopModules/NEWS/DinhKem/9079_Nganh-Y-khoa.pdf<br>- Đào tạo theo nhu cầu xã hội (theo đặt hàng ) : CTDT Y khoa<br>- Đào tạo hệ quốc tế tiếng anh : CTDT QqTe Y khoa<br>Hệ liên thông : <br>- Đào tạo liên thông hệ chính quy : CTDT lt chinh quy Y khoa<br>- Đào tạo liên thông theo nhu cầu xã hội : CTDT lt chinh quy Y khoa\nHuman: chương trìnhh đào tạo ngành y khoa"

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = [context]

metric = FaithfulnessMetric(
    threshold=0.7,
    model=custom_model_gemini,
    include_reason=True
)

test_case = LLMTestCase(
    input="Chương trình đào tạo của ngành Dược như thế nào?",
    actual_output=actual_output,
    retrieval_context=retrieval_context
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])
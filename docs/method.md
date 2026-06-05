# Phương pháp — VSDS HackAIthon 2026, Bảng C (Innovator)

Hệ thống trả lời trắc nghiệm tiếng Việt chạy hoàn toàn offline trong Docker:
đọc `/data/*.{json,csv}`, mỗi câu chọn một chữ cái, ghi `/output/pred.csv`.

**Kết quả:** baseline thuần LLM 66.52% → **82.29%** trên public leaderboard
(+15.77 điểm phần trăm), bằng nâng cấp mô hình + tool-augmentation, **không**
dùng RAG (đã đo và loại — xem §5).

---

## 1. Triết lý thiết kế

Không phải câu nào cũng cần "động não". Định tuyến theo loại câu hỏi để vừa giữ
accuracy cao vừa không phá điểm tốc độ:

```
/data/*.{json,csv}
   │  loader (chịu được JSON / CSV nhiều layout, cột thừa, BOM)
   ▼
[Router] câu có cần tính toán không?
   ├── có (LaTeX / lựa chọn số / từ khoá toán)  → [Program-of-Thought]
   │                                                Qwen viết Python (sympy)
   │                                                → sandbox chạy → lấy kết quả
   │                                                → chọn đáp án theo kết quả
   └── không (factual / đọc hiểu / suy luận)     → [LLM trả lời thẳng]
                                                    GBNF grammar ép đúng 1 chữ cái
   ▼
/output/pred.csv  (qid,answer ∈ A..K theo số lựa chọn thực tế)
```

Mọi nhánh đều có **fallback**: lỗi tool / lỗi parse → quay về câu trả lời trực
tiếp; không câu nào làm vỡ cả lô.

## 2. Mô hình

- **LLM:** `Qwen3.5-9B-Instruct` Q4_K_M (GGUF), đúng danh mục BTC (Qwen3.5 ≤ 9B).
  Chạy qua `llama-cpp-python` build CUDA; cùng một image chạy được cả GPU lẫn CPU.
- **Decoding:** `temperature=0`, `seed=42`, **GBNF grammar** `root ::= "A" | ... |
  "<chữ cái cuối>"` sinh động theo số lựa chọn từng câu → output luôn hợp lệ,
  không cần hậu xử lý mong manh.
- Không dùng mô hình embedding/rerank trong cấu hình nộp (RAG đã loại).

## 3. Program-of-Thought (đòn bẩy chính)

Mô hình 7–9B hay sai số học và suy luận nhiều bước nhưng lại viết được code đúng.
Thay vì để mô hình "nhẩm", ta để nó **viết một đoạn Python ngắn** (được dùng
`sympy`), chạy trong **subprocess cô lập** (`python -I`, timeout, chỉ tính toán),
rồi đưa kết quả lại cho mô hình chọn đáp án (grammar ép 1 chữ cái).

Vì sao Program-of-Thought chứ không phải function-calling nhiều vòng: mô hình
GGUF cỡ nhỏ phát một khối code đáng tin, nhưng **không** ổn định ở hội thoại
tool-call JSON nhiều bước (quyết định gọi → đúng format → đọc kết quả → chốt — mỗi
bước một điểm hỏng). PoT gộp lại còn 1 lần sinh + 1 lần chạy, kèm **retry có
giới hạn** (code lỗi → đưa stderr lại, sửa 1 lần) và fallback an toàn.

**Router** (precision-favoring): chỉ đẩy vào PoT khi có tín hiệu tính toán mạnh
(LaTeX, lựa chọn toàn số, từ khoá "tính/đạo hàm/tích phân/phương trình"). Câu đọc
hiểu và factual giữ ở luồng trực tiếp — đưa nhầm câu factual vào PoT tạo "số bịa"
làm hỏng đáp án (đúng cơ chế hại của RAG, §5).

## 4. Kết quả & phân rã đóng góp

Không có nhãn thật (BTC giữ kín) nên trong quá trình phát triển dùng
**dự đoán của một mô hình tham chiếu mạnh làm proxy** để đo *tỉ lệ đồng thuận*,
giúp so sánh các phiên bản mà không tốn lượt nộp. Số leaderboard là accuracy thật.

| Phiên bản | Mô hình | Phương pháp | Đồng thuận proxy | Leaderboard |
|---|---|---|---|---|
| v1 | Qwen2.5-7B | thuần | 70.41% | **66.52%** |
| v3 | Qwen2.5-7B | + RAG | 65.01% | (loại) |
| v4 | Qwen2.5-7B | + tools | 75.16% | — |
| — | Qwen3.5-9B | thuần | 76.46% | — |
| **v4** | **Qwen3.5-9B** | **+ tools** | **86.83%** | **82.29%** |

Phân rã trên Qwen3.5: nâng cấp mô hình +6.05đ proxy, **tools +10.37đ proxy**
(58 câu đổi: 53 đúng thêm, chỉ 5 hỏng). Tools là đòn bẩy lớn nhất và rất "sạch".

## 5. RAG: kết quả âm có kiểm chứng (đáng giá phần "ý tưởng")

Giả thuyết ban đầu: nhồi ngữ cảnh Wikipedia tiếng Việt sẽ giúp các câu factual.
Đã dựng RAG đầy đủ — **1,155,009 chunk** Wikipedia-VN, BGE-m3 1024d, FAISS
IndexFlatIP — và đo nghiêm túc:

- RAG luôn bật **giảm** 5.40 điểm proxy trên Qwen2.5.
- **Quét ngưỡng** (chỉ nhồi khi độ tương đồng ≥ T): **không ngưỡng nào** vượt
  baseline; tốt nhất là khi RAG gần như không kích hoạt.
- **Kiểm chứng lại trên mô hình mạnh hơn** (Qwen3.5): vẫn giảm 2.81 điểm. Mô hình
  mạnh ít bị phân tâm hơn nhưng **vẫn âm** → loại trừ giả thuyết "mô hình yếu nên
  RAG chưa phát huy".

**Nguyên nhân gốc:** corpus tổng quát hiếm khi chứa đáp án cho câu hỏi đặc thù VN
(giáo trình, luật, chính sách); retriever vẫn trả về đoạn "cùng chủ đề nhưng không
có đáp án" → kéo mô hình rời câu nó vốn đoán đúng. Đây là quyết định **data-driven**
để loại RAG, không phải theo cảm tính. Chi tiết: [`rag-experiment.md`](rag-experiment.md).

Mã RAG được giữ lại sau cờ `--rag` (mặc định tắt) để tái lập và kiểm toán; image
nộp **không** đóng gói nó.

## 6. Độ bền & khả năng tái lập (bảo vệ điểm vòng 2)

Điểm vòng 2 = BTC chạy container trên 2000 câu private, offline, phần cứng của họ.
Ba việc đã làm để bảo vệ điểm bất kể accuracy:

- **Loader chịu lỗi:** chấp nhận JSON *và* CSV ở nhiều layout (`choices` dạng
  JSON trong ô, hoặc các cột `A,B,C,D,...`), bỏ qua cột/khóa thừa (BTC có thể thêm
  `id`/`category`), đọc BOM (Excel). Đã test trên private_test giả lập cả 3 layout.
- **Xác định (deterministic):** `temperature=0`, `seed=42`, sandbox không ngẫu
  nhiên → chạy 2 lần ra **byte-identical** (đạt tiêu chí "chạy ≥ 3 lần ổn định").
- **Offline GPU/CPU:** llama-cpp build CUDA vẫn **import + chạy khi không có GPU**;
  `n_gpu_layers` tự dò (GPU → 99, CPU → 0). Một image chạy cả hai môi trường.
- Mọi câu đều có đáp án hợp lệ (fallback an toàn) — không thiếu dòng, không ký tự
  ngoài tập hợp lệ.

## 7. Tái lập

```bash
# 1. tải GGUF (một lần)
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/qwen3.5-9b
# 2. build image (đã bao gồm trọng số, không cần mạng lúc chấm)
bash scripts/docker_build.sh
# 3. chạy
docker run --rm --gpus all -v $PWD/data:/data -v $PWD/output:/output vsds-bangc:latest
```

## 8. Hướng phát triển tiếp (đã phác thảo, chưa đưa vào bản nộp)

RAG **có mục tiêu hẹp** trên nguồn có mật độ đáp án cao — văn bản luật VN (Bộ luật
Hình sự/Dân sự, luật chuyên ngành) và giáo trình Tư tưởng HCM / Triết Mác-Lênin —
kèm **định tuyến theo domain + rerank + ngưỡng**, là hướng khả thi nhất để chạm
nhóm câu factual đặc thù mà Wikipedia tổng quát không giải được. Sẽ pilot trên luật
trước (xác suất thành công cao nhất, là bài toán tra cứu thuần) và chỉ giữ nếu đo
được net-positive.

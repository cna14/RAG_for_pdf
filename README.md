# 🚀 Trợ lý AI Phân tích Tài liệu Đa phương thức

Đây là một dự án ứng dụng web AI, cho phép người dùng hỏi đáp và tương tác với các tài liệu PDF của mình. Hệ thống có khả năng xử lý đa dạng các loại file PDF, hiểu câu hỏi dưới dạng văn bản và hình ảnh, đồng thời quản lý các phiên làm việc (dự án) một cách độc lập.

Dự án này đã trải qua quá trình phát triển từ một prototype ban đầu sử dụng Streamlit (`RAGupdate++.py`) đến một ứng dụng web Client-Server hoàn chỉnh với giao diện tùy chỉnh và kiến trúc mạnh mẽ.

## ✨ Các tính năng chính

* **Xử lý PDF Toàn diện:** Tự động nhận diện và áp dụng **OCR** (Nhận dạng ký tự quang học) cho các file PDF được scan, đảm bảo trích xuất được tối đa nội dung văn bản.
* **Hỏi đáp Đa phương thức:** Người dùng có thể:
    * Đặt câu hỏi bằng văn bản.
    * Đính kèm hình ảnh làm **ngữ cảnh bổ sung** cho câu hỏi.
    * Gửi một hình ảnh chứa **câu hỏi/bài toán** để AI tự động đọc và xử lý.
* **Quản lý Đa dự án:** Cho phép tạo nhiều luồng trò chuyện (dự án) riêng biệt. Mỗi dự án có "bộ não" kiến thức độc lập, chỉ học từ các file PDF được tải lên trong dự án đó.
* **Giao diện Hiện đại & Tương tác cao:**
    * Giao diện được thiết kế theo phong cách tối giản, hiện đại.
    * Sidebar điều khiển có thể thu gọn.
    * Hỗ trợ kéo & thả file PDF để học.
    * Hỗ trợ dán ảnh trực tiếp từ clipboard (`Ctrl+V`) vào khu vực chat.
    * Đổi tên và xóa dự án ngay trên giao diện.
    * Thông báo trạng thái (loading, success, error) một cách trực quan, không dùng popup.
* **Tư duy AI Nâng cao:** Sử dụng mô hình **Gemini** kết hợp với kiến trúc **RAG** (Retrieval-Augmented Generation) và các kỹ thuật tiên tiến để đưa ra các câu trả lời được suy luận, tổng hợp thay vì chỉ trích xuất thông tin.

---

## 🏛️ Kiến trúc Hệ thống

Phiên bản cuối cùng của dự án (`RAGwithUXUI`) được xây dựng theo kiến trúc **Client-Server**, tách biệt hoàn toàn giữa giao diện người dùng và bộ não xử lý.



[Image of a client-server architecture diagram]


* **Backend (`backend_api.py`):**
    * Là một API Server được xây dựng bằng **Flask**, chịu trách nhiệm xử lý toàn bộ logic nặng.
    * **Công nghệ:** `Flask`, `LangChain`, `PyMuPDF`, `Pytesseract` (cho OCR), `ChromaDB` (cơ sở dữ liệu vector), và `google-generativeai` (để giao tiếp trực tiếp với Gemini).

* **Frontend (`index.html`, `style.css`, `script.js`):**
    * Là một giao diện web tĩnh, có thể chạy trực tiếp trên trình duyệt.
    * **Công nghệ:** `HTML5`, `CSS3`, `JavaScript` thuần, và `marked.js` để hiển thị định dạng Markdown.

---

## 🌊 Luồng hoạt động & Lý thuyết (Workflow & Theory)

Về cốt lõi, dự án này áp dụng mô hình kiến trúc **Retrieval-Augmented Generation (RAG)**. Hãy tưởng tượng AI làm một bài thi "mở sách": thay vì chỉ dựa vào kiến thức đã được huấn luyện sẵn, nó có khả năng "lật" đúng trang sách (tài liệu PDF của bạn) để tìm ra thông tin chính xác trước khi trả lời.

Quy trình này được chia thành hai giai đoạn chính: **Học tài liệu** và **Trả lời câu hỏi**.

### Giai đoạn 1: Học tài liệu (Lập chỉ mục - Indexing)
Đây là quá trình biến một file PDF thành một "bộ não" kiến thức mà máy tính có thể tìm kiếm theo ngữ nghĩa.



1.  **Thu nạp & Tiền xử lý (Ingestion & Preprocessing)**
    * **Mục tiêu:** Trích xuất toàn bộ nội dung văn bản sạch từ file PDF.
    * **Lý thuyết:** File PDF có hai loại chính: "kỹ thuật số" (văn bản được lưu dưới dạng dữ liệu) và "scan" (mỗi trang là một hình ảnh). Với loại scan, chúng ta cần công nghệ **OCR (Nhận dạng ký tự quang học)** để chuyển đổi hình ảnh chữ thành văn bản.
    * **Trong dự án:** Hàm `extract_text_from_pdf_with_ocr` sử dụng `PyMuPDF` (`fitz`) để ưu tiên lấy văn bản kỹ thuật số. Nếu không có, nó sẽ tự động kích hoạt `Tesseract` để thực hiện OCR.

2.  **Phân đoạn (Chunking)**
    * **Mục tiêu:** Chia nhỏ văn bản dài thành các đoạn có kích thước phù hợp.
    * **Lý thuyết:** Các mô hình ngôn ngữ lớn (LLM) có một giới hạn về lượng văn bản có thể xử lý cùng một lúc (context window). Việc chia nhỏ giúp AI tập trung vào những thông tin liên quan nhất.
    * **Trong dự án:** Sử dụng `RecursiveCharacterTextSplitter` để chia văn bản thành các đoạn khoảng 1500 ký tự, có gối đầu lên nhau để không làm mất ngữ cảnh giữa các đoạn.

3.  **Vector hóa & Lập chỉ mục (Embedding & Indexing)**
    * **Mục tiêu:** Chuyển đổi các đoạn văn bản thành một dạng mà máy tính có thể hiểu và so sánh ý nghĩa.
    * **Lý thuyết:** Mỗi đoạn văn bản được đưa qua một mô hình **Embedding** để chuyển thành một **vector** (một dãy số hàng trăm chiều). Vector này giống như một "tọa độ GPS" trong một không gian ngữ nghĩa khổng lồ. Các đoạn văn có ý nghĩa tương tự sẽ có "tọa độ" gần nhau.
    * **Trong dự án:** `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) được dùng để tạo vector. Các vector này sau đó được lưu trữ và lập chỉ mục trong cơ sở dữ liệu vector **`ChromaDB`**.



### Giai đoạn 2: Trả lời câu hỏi (Truy xuất & Tạo sinh)
Đây là quá trình AI sử dụng "bộ não" đã được xây dựng để tìm kiếm và trả lời câu hỏi.



1.  **Phân tích Yêu cầu & Tiền xử lý Đa phương thức:**
    * **Mục tiêu:** Xác định ý định thực sự của người dùng và chuẩn bị đầy đủ thông tin.
    * **Trong dự án:** Khi người dùng gửi yêu cầu, hàm `ask_question` trong backend sẽ thực hiện **Phân loại Ý định Thông minh**:
        * Nếu có ảnh đính kèm, nó sẽ chạy hàm `analyze_attached_image` để thực hiện cả **OCR** (lấy text) và **Phân tích Trực quan** (dùng Gemini để mô tả biểu đồ, bố cục).
        * Nó xác định **"Nhiệm vụ Chính"**: Nếu người dùng gõ câu hỏi, đó là nhiệm vụ chính. Nếu không, văn bản OCR từ ảnh sẽ là nhiệm vụ chính.

2.  **Truy xuất Thông tin (Retrieval)**
    * **Mục tiêu:** Dựa trên "Nhiệm vụ Chính", tìm ra những đoạn văn bản liên quan nhất trong cơ sở dữ liệu.
    * **Trong dự án:** Sử dụng `MultiQueryRetriever`, một kỹ thuật nâng cao giúp tạo ra nhiều biến thể của "Nhiệm vụ Chính" để tìm kiếm toàn diện hơn, lấy ra các đoạn văn bản có liên quan nhất làm ngữ cảnh.

3.  **Tổng hợp & Tạo sinh (Synthesis & Generation)**
    * **Mục tiêu:** Tạo ra một câu trả lời mạch lạc, được suy luận từ tất cả các nguồn thông tin.
    * **Trong dự án:** Một **Prompt Template** chi tiết được xây dựng, bao gồm các mục riêng biệt: `NHIỆM VỤ CHÍNH`, `NGỮ CẢNH TỪ TÀI LIỆU`, và `PHÂN TÍCH HÌNH ẢNH ĐÍNH KÈM`. Toàn bộ prompt này được gửi đến mô hình Gemini, yêu cầu nó hành động như một chuyên gia, kết hợp tất cả thông tin để tạo ra một câu trả lời hoàn toàn mới và định dạng nó theo Markdown.

---

## ⚙️ Hướng dẫn Cài đặt & Khởi chạy

### 1. Yêu cầu Tiên quyết
* **Python 3.9+**
* **Tesseract OCR Engine:** Đây là yêu cầu **bắt buộc**.
    * Truy cập trang [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) để tải và cài đặt.
    * **QUAN TRỌNG:** Trong quá trình cài đặt, hãy đảm bảo bạn đã chọn cài đặt gói ngôn ngữ **Tiếng Việt (Vietnamese)**.
    * Sau khi cài đặt, bạn cần cấu hình đường dẫn chính xác đến file `tesseract.exe` trong file `backend_api.py`.

### 2. Cài đặt Môi trường
1.  Mở terminal trong thư mục gốc của dự án (`code/`).
2.  **Tạo môi trường ảo:** `python -m venv .venv`
3.  **Kích hoạt môi trường ảo:**
    * Windows: `.\.venv\Scripts\activate`
    * macOS/Linux: `source .venv/bin/activate`
4.  **Cài đặt các thư viện:** `pip install -r requirements.txt` (Nếu bạn chưa có file này, hãy tạo nó với nội dung bên dưới).
5.  **Tạo file `.env`:** Trong thư mục `code`, tạo một file mới tên là `.env` và thêm vào đó API Key của bạn:
    ```
    GOOGLE_API_KEY="AIzaSy...your_key_here"
    ```
6.  **Cấu hình Tesseract:** Mở file `backend_api.py` (hoặc tên file backend cuối cùng của bạn) và chỉnh sửa dòng sau cho đúng với đường dẫn Tesseract trên máy của bạn:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r'E:\university\V.1\chuyendeCNTT\Tesseract\tesseract.exe'
    ```

#### Nội dung file `requirements.txt`
```
flask
flask-cors
python-dotenv
fitz
PyMuPDF
Pillow
pytesseract
langchain
langchain-community
langchain-core
langchain-google-genai
sentence-transformers
chromadb
google-generativeai
marked
```

---

## 📖 Hướng dẫn Sử dụng

Bạn có hai cách để chạy dự án: phiên bản Client-Server hoàn chỉnh (khuyến nghị) hoặc phiên bản prototype cũ.

### **Cách 1: Chạy Ứng dụng Hoàn chỉnh (Client-Server)**
Đây là phiên bản mới nhất với đầy đủ tính năng và giao diện tùy chỉnh. Bạn cần chạy **hai quy trình song song**.

#### **a. Chạy Backend Server:**
Mở một terminal, kích hoạt môi trường ảo (`.\.venv\Scripts\activate`) và chạy lệnh (hãy thay `backend_api.py` bằng tên file backend cuối cùng của bạn):
```bash
python backend_api.py
```
Hãy để yên cửa sổ terminal này. Nó là bộ não của ứng dụng đang hoạt động.

#### **b. Mở Giao diện Frontend:**
1.  Trong PyCharm, tìm file `index.html`.
2.  Di chuột vào góc trên bên phải của cửa sổ code và **nhấp vào icon của trình duyệt** (Chrome, Firefox, ...).
3.  Giao diện ứng dụng sẽ mở ra trên trình duyệt và sẵn sàng để bạn tương tác.

### **Cách 2: Chạy Prototype cũ (`RAGupdate++.py`)**
Đây là phiên bản cũ hơn được xây dựng hoàn toàn bằng Streamlit. Nó không có giao diện tùy chỉnh hay các tính năng nâng cao như quản lý đa dự án, nhưng vẫn hoạt động độc lập.

1.  Mở một terminal và kích hoạt môi trường ảo (`.\.venv\Scripts\activate`).
2.  Chạy lệnh:
    ```bash
    streamlit run RAGupdate++.py
    ```
3.  Một tab mới trên trình duyệt sẽ tự động mở ra với giao diện của ứng dụng Streamlit.
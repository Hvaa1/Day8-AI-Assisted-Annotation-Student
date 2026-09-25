# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Hoàng Việt Anh

Công cụ gán nhãn đã dùng: CVAT local, sửa pre-label bằng CVAT rồi kiểm tra lại bằng `REVIEW_LOG.csv` và `round1_diff.md`

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian với vùng đệm ở giữa vì dữ liệu video có tính tương quan cao giữa các frame liền kề. Nếu chia ngẫu nhiên, nhiều frame rất gần nhau sẽ xuất hiện đồng thời ở train và test, khiến mô hình chỉ học được “bản sao” cùng cảnh chứ không phải khái quát hóa trên cảnh mới. Khi ấy, các số đo như AP50, precision, recall sẽ bị đánh giá quá cao, vì mô hình gặp lại các tình huống tương tự đã thấy lúc train. Việc giữ khoảng cách thời gian là cách giảm rủi ro rò rỉ thông tin (data leakage) và đánh giá đúng khả năng xử lý cảnh mới trong video.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg`, mô hình khởi đầu lạnh không khớp nhãn tham chiếu ở các xe nhỏ ở xa, xe gần mép hình, xe bị che một phần và các xe ở vùng bóng tối. Nhìn chung, mô hình làm khá tốt với xe lớn và sáng hơn, nhưng bỏ sót nhiều xe trung bình và nhỏ. Điều này được phản ánh rõ qua recall theo kích thước: `R small = 0.182`, `R medium = 0.547`, `R large = 0.561`. Tức là mô hình có độ phủ rất thấp cho xe nhỏ, và vẫn chưa ổn với xe trung bình.

Một trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai là khi xe quá nhỏ, bị cắt mép ảnh hoặc chỉ còn chấm đèn, như mô tả trong `reports/BLIND_SCAN.md` cho `frame_0312.jpg`: “gần chân cầu có vài xe rất xa, chỉ còn hai chấm đèn” và “góc dưới bên phải có một xe bị mép ảnh cắt mất, chỉ còn đuôi”. Những trường hợp như vậy có thể bị bỏ qua trong nhãn tham chiếu vì không đủ rõ để xác định chắc chắn, nên không nên kết luận ngay rằng model sai khi nhãn tham chiếu cũng khó xác định.

## 3. Chiến lược chọn mẫu

Chiến lược chọn mẫu áp dụng công thức `score = W_U·U + W_A·A + W_D·D` với `W_U = 0.5`, `W_A = 0.3`, `W_D = 0.2`. Ở đây, `U` phản ánh mức độ bất định của các box khó nhất trong một frame; `A` phản ánh số box “mơ hồ” trong vùng `0.15 <= c < 0.50`; `D` phản ánh khoảng cách thời gian so với ảnh đã chọn gần nhất để tránh chọn các frame trùng nhau. Vai trò của `MIN_GAP_S` là loại bỏ các frame gần như nhau về thời điểm, vì hai cảnh gần nhau trên cùng camera hầu như lặp lại, mà gán nhãn cả hai tốn công nhưng học thêm rất ít. Như vậy, mục tiêu không phải là chọn frame “đẹp nhất” mà là chọn frame có nhiều thông tin khó và ít trùng lặp.

Trong `reports/SELECTION.md`, tôi ưu tiên các frame `frame_0182.jpg`, `frame_0369.jpg`, `frame_0380.jpg`, `frame_0326.jpg`, `frame_0331.jpg`. Những frame này nằm ở top đầu `outputs/selection_round1.csv` và đều có `score` cao, số box lớn và số box mơ hồ cao. Ví dụ:
- `frame_0182.jpg`: `score = 0.9591`, `U = 0.9182`, `n_ambiguous = 18`, `n_boxes = 28`.
- `frame_0369.jpg`: `score = 0.9324`, `n_boxes = 43`, `n_ambiguous = 16`.
- `frame_0380.jpg`: `score = 0.9170`, `n_boxes = 40`, `n_ambiguous = 15`.

Các cảnh này cho thấy model đang rất phân vân ở nhiều xe cùng lúc, nên nếu gán nhãn đúng thì có khả năng giúp mô hình học nhiều hơn so với các frame dễ quá mức.

Một frame có điểm cao nhưng không chọn là `frame_0372.jpg` (`score=0.9101`), vì nó gần với các frame `frame_0369.jpg` và `frame_0380.jpg` về thời gian và cùng thuộc vùng cảnh tương tự. Cùng một luồng giao thông lặp lại thì gán nhãn `frame_0372.jpg` chỉ là chi tiêu tương tự với lợi ích học tập thấp. Một frame có điểm trung bình nhưng đáng xem là `frame_0270.jpg` (`score=0.8878`), vì dù không phải top 5, nó vẫn có `n_boxes=35` và `n_ambiguous=14`, diễn tả một cảnh có nhiều xe nhưng cấu trúc khác với nhóm top. Vậy nên, điểm bất định không tự động chứng minh ảnh đó sẽ cải thiện mô hình; chỉ chứng minh ảnh đó “có khả năng học nhiều hơn” nếu đó là trường hợp khó nhưng không gần trùng với ảnh đã chọn.

## 4. Các vòng học chủ động (active learning)

Từ `outputs/round1_diff.md`: mức độ sửa nhãn gợi ý ở vòng 1 là:

- accepted: 139
- edited: 17
- deleted: 13
- added: 119
- accept rate: 82%

Tức là trong 12 ảnh, model gợi ý 169 box nhưng sau khi sửa còn 275 box; nghĩa là người rà đã thêm nhiều box bị thiếu và xóa nhiều box sai. Điều này cho thấy pre-label ban đầu không nhất quán và cần phải chỉnh nhiều chỗ.

Bảng so sánh các vòng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 275 | 0.652 | -0.119 | 0.982 | 0.137 | 0.240 | 0.000 | 0.115 | 0.512 |

AP50 giảm 0.119 so với cold start, và recall giảm mạnh từ 0.489 xuống 0.137. Điều này cho thấy dữ liệu train mới không làm mô hình mạnh hơn cho tập test, mà làm tăng độ sợ hãi/thiếu recall ở xe nhỏ và trung bình. Về nhóm xe, xe lớn còn giữ mức recall tương đối (`0.512`), nhưng xe nhỏ mất hẳn (`0.000`) và xe trung bình giảm rất mạnh (`0.115`).

Một ca đổi sau fine-tune có thể quan sát trên `compare_round0.jpg` và `compare_round1.jpg`: trong nhiều frame, mô hình sau train không còn phát hiện các xe ở vùng xa hoặc mép hình; nó có xu hướng “giảm số lượng detection” và chỉ giữ lại các xe rõ ràng hơn. Đây là ví dụ cho thấy mô hình tiến triển theo hướng “an toàn hơn” nhưng lại bỏ sót quá nhiều xe. Lý do có thể kiểm là vì tập train quá nhỏ và không đều về kích thước xe, nên fine-tune có thiên hướng ưu tiên các mẫu lớn, rõ hơn.

Phân biệt các nguồn chứng cứ:
- `BLIND_SCAN.md` là quan sát độc lập: tôi trước tiên đã đánh giá `frame_0312.jpg` có 19 xe và ghi ra các khu vực dễ bị model bỏ sót.
- `REVIEW_LOG.csv` là cập nhật lỗi pre-label đã sửa: ví dụ `frame_0326.jpg` “sửa vì 1 xe và thêm 1 xe”, `frame_0331.jpg` “xóa một box nhầm”, `frame_0099.jpg` và `frame_0107.jpg` “added” vì model thiếu xe thực tế.
- `metrics_round1.json` là kết quả mô hình sau train: thấy rõ recall rớt mạnh ở xe nhỏ và trung bình.

Một ca khó theo guideline là các xe rất xa hoặc nằm ở mép ảnh, vì mơ hồ, khó xác định class, có thể bị bỏ qua hoặc bị phát hiện nhầm. Đó chính là loại trường hợp mà người rà phải cẩn thận: không được giữ box mơ hồ chỉ vì model gợi ý, và cũng không được xóa quá mức các vật thể có thể là xe thực sự.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start là không cải thiện: AP50 giảm từ 0.771 xuống 0.652, recall giảm từ 0.489 xuống 0.137. Điều này cho thấy chọn mẫu và sửa nhãn đúng là cần thiết, nhưng lượng dữ liệu train hiện tại chưa đủ để làm mô hình mạnh hơn trên test set. Với dữ liệu video này, nếu chỉ có 12 ảnh và các box đã được gán lại, mô hình dễ bị bias theo đối tượng lớn và rõ hơn, nhưng lại bỏ sót phần lớn xe nhỏ và trung bình. Vì vậy tôi sẽ dừng ở vòng này nếu mục tiêu là tối ưu AP50. Nếu tiếp tục, cần đổi chiến lược theo hướng chọn các frame có nhiều xe nhỏ và vùng tối hơn, chứ không chỉ chọn frame có độ bất định cao đơn thuần.

Hai ca còn yếu hoặc bất định cho vòng sau nên là:
1. Các frame có nhiều xe nhỏ ở xa hoặc xe bị chặn bởi xe lớn, vì đây là nhóm dễ bỏ sót nhưng lại có chi phí rà nhãn cao vì phải xác định từng xe thật kỹ.
2. Các frame có nhiều xe ở mép góc hoặc vùng sáng tối lẫn nhau, vì đây là nhóm dễ tạo false positive nhưng cũng rất dễ gây ảnh gần trùng nếu chọn quà nhiều frame cùng thời điểm.

Tập kiểm thử chỉ có 20 ảnh, và theo quy tắc, các xe quá nhỏ được bỏ qua; đồng thời nhãn tham chiếu không được rà thủ công hoàn toàn, mà do mô hình tạo ra trước đó. Điều đó làm cho kết luận về AP50 chỉ là ước lượng tương đối, không phải “chân lý tuyệt đối”. Nếu AP50 giảm, bước đầu tôi sẽ kiểm tra: (1) liệu ảnh được chọn có quá gần nhau và lặp lại không, (2) có bị bỏ sót nhiều xe nhỏ/xe mép ảnh trong label không, (3) có sai lệch phân phối kích thước train so với test không, (4) có lỗi trong label hoặc một vài cảnh quá khó bị nhầm khi đánh giá không. Chỉ sau khi kiểm tra các nguyên nhân này mới nên quyết định train thêm hay dừng.

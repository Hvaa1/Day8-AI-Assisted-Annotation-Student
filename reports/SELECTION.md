# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, tôi ưu tiên năm frame sau nếu chỉ có ngân sách rà năm ảnh. Tôi chọn theo trọng số `score = 0.5·U + 0.3·A + 0.2·D` và áp dụng luật khoảng cách `MIN_GAP_S = 2.0s` để tránh chọn các frame gần nhau như ảnh trùng cùng một luồng xe. Nhóm 5 frame đầu tiên là:

1. `frame_0182.jpg` — `score=0.9591`, `t_sec=72.8s`, thứ tự 1. Đây là trường hợp cực mạnh: `U=0.9182`, `n_boxes=28`, `n_ambiguous=18`, cho thấy model rất phân vân ở nhiều xe trong cùng cảnh. Cảnh này có nhiều xe hoạt động ở các khoảng cách khác nhau, nên nhãn sửa ở đây có khả năng cải thiện recall và học được các ví dụ “rất khó”.
2. `frame_0369.jpg` — `score=0.9324`, `t_sec=147.6s`, thứ tự 2. Cảnh này có `n_boxes=43` và `n_ambiguous=16`, là một trường hợp dày đặc nhưng không quá gần nhau để trở thành ảnh trùng. Nó rất phù hợp để tăng độ bền mit cho các xe ở vùng trung bình/nặng.
3. `frame_0380.jpg` — `score=0.9170`, `t_sec=152.0s`, thứ tự 3. Cảnh này khá tương đồng với `frame_0369.jpg` nhưng cách nhau 4.4s và có nhịp xe hơi khác, nên tôi giữ cả hai vì mỗi frame mang một cấu trúc vùng sáng/tối và các xe bị che kém khác nhau. Đây là ví dụ chọn theo sự đa dạng thời gian thay vì chỉ chọn top 1.
4. `frame_0326.jpg` — `score=0.9155`, `t_sec=130.4s`, thứ tự 4. Số box `39` và `n_ambiguous=15` cho thấy cảnh tương đối kín, với nhiều xe vừa và lớn trong cùng mặt phẳng. Đây là ví dụ tốt để kiểm tra box gợi ý khi xe ở góc và có phần che khuất.
5. `frame_0331.jpg` — `score=0.9154`, `t_sec=132.4s`, thứ tự 5. Mặc dù gần với `frame_0330.jpg` (chênh 0.4s, cùng vùng thời gian), tôi vẫn ưu tiên `frame_0331.jpg` vì `n_boxes=47` và `n_ambiguous=18` cao hơn rõ rệt. Đồng thời `frame_0330.jpg` có `score=0.8899` và đã gần trùng, nên không đáng chi tiêu thêm khi có giới hạn 5 ảnh.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

- `frame_0182.jpg` (thứ 1, 0.9591): có `U=0.9182`, `A=1.0`, `n_ambiguous=18`; nằm trong ảnh contact sheet cùng một vùng đường cao tốc, có nhiều xe sáng và nhiều đối tượng có độ lẫn lộn cao. Đây là ví dụ đầu tiên cho thấy model đang đánh dấu các xe “khó”, chứ không chỉ là những xe dễ.
- `frame_0369.jpg` (thứ 2, 0.9324): `n_boxes=43`, `n_ambiguous=16`; trên contact sheet là cảnh sáu gói xe dày và nhiều xe ở khoảng cách trung bình, rất phù hợp để kiểm tra xem model có bỏ sót xe bị che hoặc box quá lớn không.
- `frame_0099.jpg` (thứ 8, 0.9063): dù không đứng trong top 5 nhưng vẫn thuộc lô 12 ảnh model chọn. `U=0.9460` rất cao và `A=1.0` theo chuẩn hóa, cho thấy bối cảnh có nhiều ôm ngực xe và độ phân vân lớn, nên nếu còn ngân sách thêm, đây cũng là lựa chọn đáng xét.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

- `frame_0372.jpg` có `score=0.9101`, cao hơn nhiều frame ở giữa bảng xếp hạng, nhưng không được chọn vì nó nằm rất gần `frame_0369.jpg` và `frame_0380.jpg` về thời gian (`147.6s`, `148.8s`, `152.0s`), tức là hiện tượng giao thông tương tự lặp lại. Với ngân sách 5 ảnh, chọn `frame_0372` đồng nghĩa với chi tiêu cho ảnh gần trùng, trong khi `frame_0326` và `frame_0331` mang các cấu hình xe khác nhau và dễ tạo ra thông tin học tập lớn hơn.
- Một frame có điểm thấp nhưng vẫn đáng xem là `frame_0270.jpg` (`score=0.8878`, hạng 13), vì dù điểm không quá cao, nó có `n_boxes=35` và `n_ambiguous=14`; nếu có thời gian, cảnh này có thể kiểm tra những xe ở góc trái/phải và ô tô phông tối, nơi model dễ nhầm lớp hoặc quên detections.

Điều phép chọn này chưa chứng minh về chất lượng mô hình: khuôn khổ này chỉ tối ưu hóa “thông tin học tập” của các frame khó, chứ không phải xác nhận rằng các ảnh này chắc chắn sẽ làm tăng AP50. Chỉ số `score` phản ánh mức độ bất định và độ không giống nhau của cảnh, không phải độ chính xác thực tế của nhãn. Ngoài ra, nếu một frame có nhiều xe nhưng gần trùng với frame khác, chi phí rà nhãn sẽ lớn hơn lợi ích học tập. Vì vậy, lựa chọn này là quyết định chi phí–lợi ích hợp lý cho vòng đầu, chứ chưa phải bằng chứng mô hình đã được cải thiện hay không.

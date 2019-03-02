# lab05_LeNguyenSonNguyen_ChuyenDeComputerVision
Nhận diện gương mặt

Report:

1/ Về CSDL hình ảnh: có 23 thư mục, nhưng chỉ có 16 thư mục có hình mẫu (tính tới thời điểm download). Chất lượng hình ảnh không đều về kích thước, độ rõ nét, ánh sáng,... Số lượng hình ảnh cho mỗi gương mặt thường từ 1-4 hình.

2/ Áp dụng 2 hướng tiếp cận:
    2.1: mô hình theo tutorial tại trang chủ tensorflow hướng dẫn về image recognition [https://www.tensorflow.org/tutorials/images/hub_with_keras#top_of_page], lúc sử dụng model Sequential trong Keras, các lớp layer lấy tại google theo đường link [https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2], với thiết lập epoch 10, steps per epoch 2 (các thông số có thể điều chỉnh qua lại): cho kết quả nhận diện được 2 gương mặt, tối đa là 3 (trong quá trình chạy model nhiều lần).
    2.2: mô hình convolutional network, với các ConvD2 tự điều chỉnh, với thiết lập epoch 10, steps per epoch 5, chỉ cho kết quả nhận diện được tối đa 2 gương mặt (đã chạy model nhiều lần).
    
Nhận xét:
    1/ Cả hai mô hình đều cho kết quả thấp 2 trên 16
    2/ Số lượng hình dùng train ít, chất lượng thấp
    3/ Cần thời gian để optimize các thông số
        

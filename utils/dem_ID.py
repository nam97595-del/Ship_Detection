import pandas as pd
import os

def count_tracked_objects(gt_file_path, pred_file_path):
    if not os.path.exists(gt_file_path) or not os.path.exists(pred_file_path):
        print("Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn!")
        return

    df_gt = pd.read_csv(gt_file_path, header=None)
    gt_total_ids = len(df_gt[1].unique())

    df_pred = pd.read_csv(pred_file_path, header=None)
    pred_total_ids = len(df_pred[1].unique())

    print(f"=== KẾT QUẢ ĐẾM OBJECT ID ===")
    print(f"File Ground Truth : {gt_file_path}")
    print(f"-> Tổng số đối tượng thực tế: {gt_total_ids}")
    print("-" * 30)
    print(f"File Prediction   : {pred_file_path}")
    print(f"-> Tổng số đối tượng track được: {pred_total_ids}")
    print("=" * 30)
    
    if pred_total_ids > gt_total_ids:
        print(f"Cảnh báo: Thuật toán đang track dư {pred_total_ids - gt_total_ids} đối tượng.")
        print("Nguyên nhân phổ biến: Có nhiều ID Switches (đứt gãy vết tạo ID mới) hoặc False Positives (nhận diện nhầm nhiễu).")
    elif pred_total_ids < gt_total_ids:
        print(f"Cảnh báo: Thuật toán đang track thiếu {gt_total_ids - pred_total_ids} đối tượng so với thực tế.")
        print("Nguyên nhân phổ biến: Mô hình bị bỏ sót (False Negatives) các đối tượng khó/quá nhỏ.")
    else:
        print("Tuyệt vời! Số lượng đối tượng track được khớp hoàn toàn với số lượng thực tế.")

file_nhan_goc = r"E:\Lab Nghien Cuu\MOTA\mota-yolo12x-bytetrack\1.1(2).txt"
file_du_doan = r"E:\Lab Nghien Cuu\MOTA\mota-yolo12x-bytetrack\pred_1.1(2)_20262027_202059.txt"

count_tracked_objects(file_nhan_goc, file_du_doan)
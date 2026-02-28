import motmetrics as mm
import pandas as pd
import os
import time

timestamp = time.strftime("%Y%m%d_%H%M%S")

def evaluate_tracking(gt_file, pred_file):
    print(f"Đang đối chiếu Ground Truth ({gt_file}) và Prediction ({pred_file})...")
    
    gt_filename = os.path.basename(gt_file)
    pred_filename = os.path.basename(pred_file)
    gt_basename = gt_filename.replace('.txt', '')

    try:
        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
        pred = mm.io.loadtxt(pred_file, fmt="mot15-2D")
    except Exception as e:
        print(f"Lỗi nạp file. Hãy kiểm tra lại định dạng file: {e}")
        return 
    
    print("🔹 Đang tính toán phép khớp (Matching)...")
    # Tính toán ma trận khoảng cách dựa trên IoU
    # distth=0.5: Box máy tính vẽ trùng 50% mới tính đúng
    acc = mm.utils.compare_to_groundtruth(gt, pred, 'iou', distth=0.5)

    # Chọn các chỉ số (metrics)
    mh = mm.metrics.create()
    metrics = [
        'num_frames', 'mota', 'idf1', 'idp', 'idr', 
        'num_switches', 'num_false_positives', 'num_misses'
    ]
    
    summary = mh.compute(acc, metrics=metrics, name=gt_basename)

    # Hiển thị kết quả
    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    print("\n" + "="*60)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG TRACKING (MOTA) 🏆")
    print("="*60)
    print(str_summary)
    print("="*60)

    # --- LƯU FILE ---
    csv_filename = f"MOTA_results_{gt_basename}_{timestamp}.csv"
    txt_filename = f"MOTA_results_{gt_basename}_{timestamp}.txt"
    
    csv_out_path = os.path.join(os.path.dirname(gt_file), csv_filename)
    summary_export = summary.rename(columns=mm.io.motchallenge_metric_names)
    summary_export.to_csv(csv_out_path)

    txt_out_path = os.path.join(os.path.dirname(gt_file), txt_filename)
    with open(txt_out_path, "w", encoding="utf-8") as f:
        f.write("BẢNG KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG TRACKING (MOTA)\n")
        f.write("="*60 + "\n")
        f.write(f"File Ground Truth : {gt_filename}\n")
        f.write(f"File Prediction   : {pred_filename}\n")
        f.write("-" * 60 + "\n")
        f.write(str_summary + "\n")
        f.write("="*60 + "\n")

    print(f"\n✅ Đã lưu kết quả ra file Excel (CSV): {csv_out_path}")
    print(f"✅ Đã lưu kết quả ra file Text (TXT): {txt_out_path}")


if __name__ == "__main__":
    mot_dir = os.path.join(os.getcwd(), "utils")
    GT_PATH = os.path.join(mot_dir, "DJI_0430(2).txt")   # File gán nhãn bằng DarkLabel
    PRED_PATH = os.path.join(mot_dir, "pred_0430_20262027_202059.txt") # File yolo_engine.py xuất ra
    
    print(GT_PATH + "\n" + PRED_PATH)
    if os.path.exists(GT_PATH) and os.path.exists(PRED_PATH):
        evaluate_tracking(GT_PATH, PRED_PATH)
    else:
        print("❌ Không tìm thấy file gt.txt hoặc pred.txt trong thư mục hiện tại!")
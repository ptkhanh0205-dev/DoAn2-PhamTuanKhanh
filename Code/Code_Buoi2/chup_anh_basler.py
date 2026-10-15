#!/usr/bin/env python3
"""
Chụp ảnh từ camera Basler acA2500-14gm (GigE) bằng pypylon và lưu ra file PNG.

Cài đặt (Windows hoặc Linux, Python 3.9 - 3.14):
    pip install pypylon opencv-python numpy

Cách chạy:
    python chup_anh_basler.py --list                  # liệt kê camera tìm thấy
    python chup_anh_basler.py                         # chụp 1 ảnh từ camera thật
    python chup_anh_basler.py --exposure 8000 --show  # phơi sáng 8000 µs, hiện ảnh
    python chup_anh_basler.py --emu                   # chạy thử bằng camera giả lập (chưa có camera)
    python chup_anh_basler.py --emu --image dongxu.png  # camera giả lập phát ảnh của bạn

Viết cho pypylon 26.6 trở lên (cú pháp camera.X.Value / TrySetValue).
"""
import argparse
import os
import sys

# Số serial in trên nhãn camera của lab (S/N: 21955418).
LAB_CAMERA_SERIAL = "21955418"


def parse_args():
    p = argparse.ArgumentParser(description="Chụp ảnh camera Basler bằng pypylon")
    p.add_argument("--list", action="store_true", help="chỉ liệt kê camera rồi thoát")
    p.add_argument("--emu", action="store_true", help="dùng camera giả lập của pylon")
    p.add_argument("--image", help="(với --emu) file ảnh để camera giả lập phát ra")
    p.add_argument("--serial", default=LAB_CAMERA_SERIAL, help="số serial camera cần mở")
    p.add_argument("--exposure", type=float, default=10000.0, help="thời gian phơi sáng, µs")
    p.add_argument("--jumbo", action="store_true",
                   help="đặt gói tin 8192 byte (chỉ dùng khi card mạng đã bật Jumbo Frame)")
    p.add_argument("--out", default="anh_chup.png", help="tên file ảnh lưu ra")
    p.add_argument("--show", action="store_true", help="hiện ảnh bằng OpenCV")
    return p.parse_args()


def list_devices(pylon):
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    print(f"Tìm thấy {len(devices)} camera:")
    for d in devices:
        try:
            ip = d.GetIpAddress() if d.IsIpAddressAvailable() else "-"
        except Exception:  # camera giả lập / USB không có IP
            ip = "-"
        print(f"  {d.GetModelName():<20} S/N {d.GetSerialNumber():<10} IP {ip}")
    return devices


def open_camera(pylon, args):
    if args.emu:
        return pylon.InstantCamera(pylon.FirstFound)
    # Ưu tiên đúng camera của lab theo số serial, nếu không có thì lấy camera đầu tiên.
    try:
        return pylon.InstantCamera({"SerialNumber": args.serial}, pylon.FirstFound)
    except Exception:  # không có camera mang số serial này
        print(f"Không thấy camera S/N {args.serial}, dùng camera đầu tiên tìm được.")
        return pylon.InstantCamera(pylon.FirstFound)


def configure(camera, args):
    # Ảnh đen trắng 8 bit: mỗi pixel 0 (đen) đến 255 (trắng).
    camera.PixelFormat.TrySetValue("Mono8")

    # Tắt tự động để các lần chụp giống hệt nhau (quan trọng khi đo).
    camera.ExposureAuto.TrySetValue("Off")
    camera.GainAuto.TrySetValue("Off")

    # acA2500-14gm (ace Classic) dùng tên ExposureTimeAbs; camera đời mới dùng ExposureTime.
    # TrySetValue bỏ qua tham số không có, nên gọi cả hai là an toàn.
    if not camera.ExposureTime.TrySetValue(args.exposure):
        camera.ExposureTimeAbs.TrySetValue(args.exposure)

    # Lấy hết vùng ảnh (2592 x 1944 với camera thật).
    camera.OffsetX.TrySetValue(0)
    camera.OffsetY.TrySetValue(0)
    camera.Width.TrySetToMaximum()
    camera.Height.TrySetToMaximum()

    if args.jumbo:
        camera.GevSCPSPacketSize.TrySetValue(8192)

    if args.emu and args.image:
        camera.TestImageSelector.TrySetValue("Off")
        camera.ImageFileMode.TrySetValue("On")
        camera.ImageFilename.TrySetValue(os.path.abspath(args.image))


def main():
    args = parse_args()
    if args.emu:
        # Phải đặt trước khi import pylon: tạo 1 camera giả lập.
        os.environ["PYLON_CAMEMU"] = "1"

    from pypylon import pylon
    import cv2

    devices = list_devices(pylon)
    if args.list:
        return 0
    if len(devices) == 0:
        print("Không có camera. Kiểm tra dây mạng, nguồn PoE, địa chỉ IP, tường lửa.")
        return 1

    with open_camera(pylon, args) as camera:
        info = camera.DeviceInfo
        print(f"Đang dùng: {info.GetModelName()} (S/N {info.GetSerialNumber()})")
        configure(camera, args)

        # GrabOne: chụp đúng 1 ảnh, chờ tối đa 5000 ms.
        with camera.GrabOne(5000) as result:
            if not result.GrabSucceeded():
                print(f"Chụp lỗi: {result.ErrorCode:#x} {result.ErrorDescription}")
                return 1
            img = result.Array  # mảng numpy, kích thước (cao, rộng)

        print(f"Ảnh {img.shape[1]} x {img.shape[0]} px, "
              f"độ sáng min/trung bình/max = {img.min()}/{img.mean():.1f}/{img.max()}")
        cv2.imwrite(args.out, img)
        print(f"Đã lưu {args.out}")

        if args.show:
            small = cv2.resize(img, None, fx=0.4, fy=0.4)  # thu nhỏ cho vừa màn hình
            cv2.imshow("Basler", small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

import cv2 as cv
import sm


def main():
    img = cv.imread('document.JPG')
    scanned_img = sm.ScannerDocument(img)
    bw = sm.ConvertBW(scanned_img)
    cv.imwrite('process_images/scanned_document_BW.jpg', bw)


if __name__ == "__main__":
    main()

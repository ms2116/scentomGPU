import cv2
inputfile_org = "20221116191643_video.webm"
ffmpeg_prefile = "tmp_20221116191643_video.webm"
resultfile = "result_tmp_20221116191643_video.webm"

cap = cv2.VideoCapture(inputfile_org)
print(cap.get(cv2.CAP_PROP_FPS))
cap.set(cv2.CAP_PROP_FPS, 50)
print(cap.get(cv2.CAP_PROP_FPS))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1)
    cv2.imshow('test', frame)
    if key == ord('q'):
        break

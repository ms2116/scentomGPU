from fastapi import FastAPI
import uvicorn
from datetime import datetime
import os
import sys
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch import nn
from threading import Thread
from tqdm import tqdm
from PIL import Image
import numpy as np
import json

from pydantic import BaseModel
from typing import List, Optional

from BackgroundMattingV2.dataset import VideoDataset, ZipDataset
from BackgroundMattingV2.dataset import augmentation as A
from BackgroundMattingV2.model import MattingBase, MattingRefine
import cv2

import ffmpeg
import io
import time
import subprocess


app = FastAPI()


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


STATIC_DIR ='static/'
BASE_DIR = "static/upload/DB/"

user_name = 'yth' # 이부분 데이터베이스로 변경

USER_DIR = f'{user_name}/'
os.makedirs(USER_DIR, exist_ok=True)
TEMP_DIR = 'temp/'
os.makedirs(USER_DIR+TEMP_DIR, exist_ok=True)

UPLOAD_DIR = os.path.join(BASE_DIR, USER_DIR)
IMG_DIR = os.path.join(STATIC_DIR,'images/')


save_dir = os.path.join(UPLOAD_DIR, TEMP_DIR)
createDirectory(save_dir)



class Bgdata(BaseModel):
    bgjson: str
    # path_list: Optional[str] = None
    # org_video_path: Optional[str] = None
    {
        "bgjson": "./",
    }




@app.post('/test')
# async def bg_test(Bgdata: Bgdata):
async def bg_test(bgdata: Bgdata):
    print(bgdata.bgjson)
    print(type(bgdata.bgjson))
    bg_json = json.loads(bgdata.bgjson)
    print(type(bg_json))

    video_json = bg_json["video_json"]
    framevideofilename = bg_json["framevideofilename"]
    org_video = bg_json["org_video"]
    print(video_json)
    print(framevideofilename)
    print(org_video)

    org_video = "C:/Users/82104/PycharmProjects/musicin_gpu_setting/20221116191643_video.webm"

    ############ ffmpeg 로 스트리밍 데이터 변환해야함 ##############

from datetime import datetime
from fastapi import APIRouter
from starlette.responses import Response
import os
from fastapi import UploadFile
import sys
import shutil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch import nn
from threading import Thread
from tqdm import tqdm
from PIL import Image
import numpy as np
import json


# backgroundmattingv2 모듈
from BackgroundMattingV2.inference_utils import HomographicAlignment
from BackgroundMattingV2.dataset import VideoDataset, ZipDataset
from BackgroundMattingV2.dataset import augmentation as A
from BackgroundMattingV2.model import MattingBase, MattingRefine
import cv2

# ffmpeg 모듈
import ffmpeg
import io
import time
import subprocess
from typing import List
import pytranscoder
# requests 를 위한
import asyncio


@app.get("/")
def index():
    """
    ELB 상태 체크용 API
    :return:
    """
    current_time = datetime.utcnow()
    return Response(f"Notification API (UTC: {current_time.strftime('%Y.%m.%d %H:%M:%S')})")


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")




@app.post('/remove_background2/bg')
async def remove_background_bg(files: List[UploadFile]):

    currentTime = datetime.now().strftime("%Y%m%d%H%M%S")

    # 받은 파일 일단 저장하기
    for file in files:
        save_filename = f'{currentTime}_{file.filename}'
        save_location_name = os.path.join(save_dir, save_filename)
        if file.filename.endswith(".webm") or file.filename.endswith(".mp4"):
            with open(save_location_name, "wb+") as file_object:
                # shutil.copyfileobj(file.file, file_object) # 둘다 가능
                file_object.write(file.file.read()) # 둘다 가능
            org_video = save_location_name
            org_video2 = await file.read()
            framevideofilename = f'{currentTime}_{file.filename}'
        elif file.filename.endswith(".json"):
            test = await file.read()
            video_json = json.loads(test)
    setting_dict = dict(video_json)
    framerate = 24 # 수정가능
    ffmpeg_out = os.path.join(save_dir, f"tmp_{framevideofilename}")

    print('ffmpegout', ffmpeg_out)
    (
        ffmpeg
        .input(org_video)
        .filter('fps', fps=framerate, round='up')
        .output(ffmpeg_out, crf=22)
        .run()
    )
    print("ffmpeg 처리끝 ")

    ######## 영상 길이 및 배경 제거 포지션 확인 ################

    duration = float(setting_dict["vidDuration"])
    bgrposition = float(setting_dict["bgTime"])
    start_position = float(setting_dict["startTime"])
    end_position = float(setting_dict["endTime"])

# ###### 프레임 넘버 확인 및 배경 따로 저장 #############
    def bgr_save_search(video_src, outdir, bgrposition, duration, save_name):
        tmpcap = cv2.VideoCapture(video_src)
        count = tmpcap.get(cv2.CAP_PROP_FRAME_COUNT)

        strat_frame_num = int((start_position / duration) * count)
        end_frame_num = int((end_position / duration) * count)
        bgr_frame_num = int((bgrposition / duration) * count)

        tmpcap.set(cv2.CAP_PROP_POS_FRAMES, bgr_frame_num)
        _, bgr_frame = tmpcap.read()
        cv2.imwrite(save_name, bgr_frame)
        return strat_frame_num, end_frame_num

    bg_save_dir = os.path.join(UPLOAD_DIR, TEMP_DIR)
    bg_save_name = os.path.join(bg_save_dir, f'{currentTime}_tmp.jpg')

    strat_frame_num, end_frame_num = bgr_save_search(ffmpeg_out, os.path.join(UPLOAD_DIR, TEMP_DIR), bgrposition, duration, bg_save_name)
    print(strat_frame_num, end_frame_num)

    video_src = ffmpeg_out
    video_bgr = bg_save_name
    print(video_src, video_bgr)


    # currentTime = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    ################### 이부분 배경영상 처리 파트###################

    class VideoWriter:
        def __init__(self, path, frame_rate, width, height):
            self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'vp09'), frame_rate, (width, height))

        def add_batch(self, frames):
            frames = frames.mul(255).byte()
            frames = frames.cpu().permute(0, 2, 3, 1).numpy()
            frame = frames[0]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)

    def VideoWriterAlpha(frames):
        frames = frames.mul(255).byte()
        frames = frames.cpu().permute(0, 2, 3, 1).numpy()
        frame = frames[0]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        return frame

    # --------------- Main ---------------
    def making_webm(video_src, video_bgr, com=None, pha=None, save_name='result', video_resize=None, strat_frame_num=None, end_frame_num=None):
        # 모델 파라미터
        device = torch.device('cuda')
        model_backbone = "mobilenetv2"
        model_backbone_scale = 0.25
        model_checkpoint = r"C:\Users\82104\PycharmProjects\musicin_last\model_pth\pytorch_mobilenetv2.pth"
        model_refine_mode = 'sampling'
        model_refine_sample_pixels = 80_000
        model_refine_threshold = 0.7
        model_refine_kernel_size = 3

        # 모델 설정 파트
        model_type = 'mattingrefine'
        model = MattingRefine(
            model_backbone,
            model_backbone_scale,
            model_refine_mode,
            model_refine_sample_pixels,
            model_refine_threshold,
            model_refine_kernel_size)

        model = model.to(device).eval()
        model.load_state_dict(torch.load(model_checkpoint, map_location=device), strict=False)

        vid = VideoDataset(video_src)
        bgr = [Image.open(video_bgr).convert('RGB')]

        framerate = vid.frame_rate
        vidcount = vid.frame_count

        # start_frame = int((float(setting_dict["startTime"])/float(setting_dict["vidDuration"]))*vidcount)
        # end_frame = int((float(setting_dict["endTime"])/float(setting_dict["vidDuration"]))*vidcount)

        vid = VideoDataset(video_src)[strat_frame_num:end_frame_num]

        dataset = ZipDataset([vid, bgr], transforms=A.PairCompose([
            # A.PairApply(nn.Identity()),
            A.PairApply(T.Resize(video_resize[::-1])),
            A.PairApply(nn.Identity()),
            A.PairApply(T.ToTensor())
        ]))

        h = video_resize[1] if video_resize else vid.height
        w = video_resize[0] if video_resize else vid.height

        png_list = []

        #배경제거 mp4 만들기
        com_writer = VideoWriter(os.path.join(output_dir, 'mp4_test_com.mp4'), framerate, w, h)

        with torch.no_grad():
            ############## webm 파일 저장을위한 process 설정 ####################
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgra', s='{}x{}'.format(w, h), r=framerate)
                .output(save_name, pix_fmt='yuva420p', vcodec='libvpx-vp9', r=framerate, crf=22)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            ############### webm 파일 저장을위한 process 설정 ####################

            ############### 배경 제거 시작 ####################
            for frnum, input_batch in enumerate(tqdm(DataLoader(dataset, batch_size=1, pin_memory=True))):
                src, bgr = input_batch

                # 배경색 설정 파트
                tgt_bgr_green = torch.tensor([0, 255 / 255, 0], device=device).view(1, 3, 1, 1)  # 배경그린
                tgt_bgr_white = torch.tensor([1, 1, 1], device=device).view(1, 3, 1, 1)
                tgt_bgr_black = torch.tensor([0, 0, 0], device=device).view(1, 3, 1, 1)
                # 배경색 설정 끝

                src = src.to(device, non_blocking=True)
                bgr = bgr.to(device, non_blocking=True)

                pha, fgr, _, _, _, _ = model(src, bgr)

                ################# 배경제거 frame 만들기/ 배경색 선택 #################

                com = src * pha  # orginal

                ################# png 리스트 만들기
                com_alpha = torch.cat((com, pha), dim=1)  # rgba frame 저장
                rgba_frame = VideoWriterAlpha(com_alpha)  # rgba 값으로 저장
                com_writer.add_batch(com) # 배경제거 mp4 만들기
                # png_list.append(rgba_frame) # 굳이 리스트 만들 필요 없어짐

        ############ webm 파일 저장구간 ##################
                process.stdin.write(
                    rgba_frame
                    .astype(np.uint8)
                    .tobytes()
                )

            process.stdin.close()
            process.wait()
            ############ webm 파일 저장구간 ##################
        result = png_list
        return result, framerate

    ### 배경제거 파라미터 설정 #####
    # video_src = r"C:\Users\82104\PycharmProjects\1.prototype\hi.mp4"
    # video_bgr = r"C:\Users\82104\PycharmProjects\musicin_backend_setting\app\DB\videos\1666944627_yth.jpg"


    output_dir = f"./static/videos/{user_name}"

    video_resize = (720, 1280)
    save_name = f'{output_dir}/result_{framevideofilename}'
    ### 배경제거 함수 실행 #############
    png_list_result, framerate = making_webm(video_src,
                                            video_bgr,
                                            save_name=save_name,
                                            video_resize=video_resize,
                                            strat_frame_num=strat_frame_num,
                                            end_frame_num=end_frame_num,
                                            )
    return "editor"


# ./configure --enable-libvpx --enable-libopus --enable-vp9




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8050, reload=True)

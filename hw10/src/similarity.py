import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import warnings
from os import walk
from io import BytesIO
from PIL import Image
import zipfile
from tqdm import tqdm

warnings.filterwarnings("ignore")

msecount = 0
ssimcount = 0
lpipscount=0
msemin=100000
ssimmax=-1
lpipsmin=100000
student1=''
student2=''
student3=''
imgcount=0

mypath = "./test"
fp = open("result.txt", "w")
loss_fn = lpips.LPIPS(net='alex')

for root, dirs, files in walk(mypath): 
    for file in files:
        print("*****************************\n","檔案：",file)
        fp.write("*****************************\n"+"檔案："+file+"\n")
        imgcount = 0
        msecount=0
        ssimcount = 0
        lpipscount = 0
        zipfile_path = './test/{}'.format(file)
        with zipfile.ZipFile(zipfile_path,mode='r') as zfile:
            words = 3000
            imgcount = len(zfile.namelist()[:words])
            for name in tqdm(zfile.namelist()[:words]):
                for i in range(len(name)):
                    if '/' in name:
                        if name[i] == '/':
                            gt = cv2.imread("./myfont/111598069_new/{}".format(name[i+1:]), cv2.IMREAD_GRAYSCALE)    
                    else:
                        gt = cv2.imread("./myfont/111598069_new/{}".format(name), cv2.IMREAD_GRAYSCALE)
                        
                if gt is not None:
                    gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float() / 255.0
                
                if ".png" in name :
                    img = np.asarray(Image.open(BytesIO(zfile.read(name))).convert('L')) #轉灰階
                    gradient = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0    
                else:
                    imgcount = imgcount - 1
                    
                if len(img) != 300 or len(img[0]) != 300:
                    imgcount = imgcount - 1
                
                if gt is not None and img is not None and len(img)==300 and len(img[0]) == 300:
                    mse = np.mean((gt.squeeze().numpy() - gradient.squeeze().numpy()) ** 2)
                    ssim_score = ssim(gt.squeeze().numpy(), gradient.squeeze().numpy(), win_size=7)
                    lpips_distance = loss_fn(gt, gradient)
                    msecount = msecount + mse
                    ssimcount = ssimcount + ssim_score
                    lpipscount = lpipscount + lpips_distance.item()
            
            mseavg = msecount/imgcount
            ssimavg = ssimcount/imgcount
            lpipsavg = lpipscount/imgcount
            print("imgcount = {}\n".format(imgcount)+
                  "MSE avg:"+"{:.6f}\n".format(mseavg)+
                  "SSIM avg:"+"{:.6f}\n".format(ssimavg)+
                  "LPIPS avg:"+"{:.6f}".format(lpipsavg))
            fp.write("imgcount = {}\n".format(imgcount)+
                     "MSE avg:"+"{:.6f}\n".format(mseavg)+
                     "SSIM avg:"+"{:.6f}\n".format(ssimavg)+
                     "LPIPS avg:"+"{:.6f}\n".format(lpipsavg))
            
            if(msemin > mseavg):
                msemin = mseavg
                student1 = file
            if(ssimmax < ssimavg):
                ssimmax = ssimavg
                student2 = file
            if(lpipsmin > lpipsavg):
                lpipsmin = lpipsavg
                student3 = file

print("############finish############\n"+
      "MSE 最像的同學:"+student1+"  MSE: "+"{:.6f}\n".format(msemin)+
      "SSIM 最像的同學:"+student2+"  SSIM: "+"{:.6f}\n".format(ssimmax)+
      "LPIPS 最像的同學:"+student3+"  LPIPS: "+"{:.6f}".format(lpipsmin))
fp.write("############finish############\n"+
         "MSE 最像的同學:"+student1+"  MSE: "+"{:.6f}\n".format(msemin)+
         "SSIM 最像的同學:"+student2+"  SSIM: "+"{:.6f}\n".format(ssimmax)+
         "LPIPS 最像的同學:"+student3+"  LPIPS: "+"{:.6f}".format(lpipsmin))
fp.close()


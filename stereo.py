import cv2
import numpy as np
import sys
from mylib import read

#Find the position of the light source on the mirror ball
def LightSpecularPoint(imgs, img_mask):
    ret, img_mask = cv2.threshold(cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    cog = []
    for img in imgs:
        img = cv2.bitwise_and(img_mask, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        ret, img_bin = cv2.threshold(img, np.max(img)-5, 255, cv2.THRESH_BINARY)
        label = cv2.connectedComponentsWithStats(img_bin)
        tmp = np.delete(label[3], 0, 0)[0]
        cog.append(tmp)
    return cog

#Find the normal image(img_normal) of the given sphere image(img_sphere)
def SphereNormalMap(img_sphere):
    ret, img_bin = cv2.threshold(cv2.cvtColor(img_sphere, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    label = cv2.connectedComponentsWithStats(img_bin)
    rec = np.delete(label[2], 0, 0)[0]
    cog = np.delete(label[3], 0, 0)[0]
    a = cog[0]
    b = cog[1] 
    c = r = (rec[2]+rec[3])/4
    img_x = np.fromfunction(lambda y, x: 2*(x-a), img_bin.shape, dtype=np.float32)
    img_y = np.fromfunction(lambda y, x: 2*(y-b), img_bin.shape, dtype=np.float32)
    img_z = np.fromfunction(lambda y, x: 2*((abs(r**2-(x-a)**2-(y-b)**2))**0.5+c - c), img_bin.shape, dtype=np.float32)
    length = np.sqrt(np.square(img_x) + np.square(img_y) + np.square(img_z))
    img_normal = cv2.merge([img_x, -img_y, img_z]) / cv2.merge([length, length, length])
    img_normal = cv2.bitwise_and(img_normal, img_normal, mask=img_bin)
    img_normal = img_normal*0.5+0.5
    img_normal = np.clip(img_normal, 0.0, 1.0)
    img_normal = (img_normal*255).astype(np.uint8)
    img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)#B-z,G-y,R-x
    return img_normal

#Find the position of the light source
def LightSourceDirection(points, img_normal):
    L_list = []
    for p in points:
        x = int(p[0])
        y = int(p[1])
        n = img_normal[y][x]/255*2 - 1
        v = np.array([1.0, 0.0, 0.0])
        s = (2*np.dot(n.T, v))*n - v
        L_list.append(s)
    return L_list

#Obtain normal image(img_normal) and albedo image(img_albedo) from light source direction(L_list) and captured image(imgs)
def PMS(imgs, L_list):
    L = np.array(L_list)
    Lt = L.T
    h, w = (imgs[0].shape)[:2]

    img_normal = np.zeros((h, w, 3))
    img_albedo = np.zeros((h, w, 3))
    I = np.zeros((len(L_list), 3))
    for x in range(w):
        for y in range(h):
            for i in range(len(imgs)):
                I[i] = imgs[i][y][x]
            tmp1 = np.linalg.inv(np.dot(Lt, L))
            tmp2 = np.dot(Lt, I)
            G = np.dot(tmp1, tmp2).T
            rho = np.linalg.norm(G, axis=1)
            img_albedo[y][x] = rho

            G_gray = G[0]*0.0722+G[1]*0.7152+G[2]*0.2126
            Gnorm = np.linalg.norm(G_gray)
            if Gnorm==0:
                continue
            img_normal[y][x] = G_gray/Gnorm

    img_normal = ((img_normal*0.5 + 0.5)*255).astype(np.uint8)
    img_albedo = (img_albedo/np.max(img_albedo)*255).astype(np.uint8)
    return img_normal, img_albedo

def main():
    path_chrome = "chrome"
    path_src = "obj"
    
    img_chromes = read.imread(path_chrome + "/target/*")
    mask_chrome = read.imread(path_chrome + "/mask/*")[0]
    print(path_chrome + " : " + str(len(img_chromes)) + " images")
    
    img_srcs = read.imread(path_src + "/target/*")
    mask_src = read.imread(path_src + "/mask/*")[0]
    print(path_src + " : " + str(len(img_srcs)) + " images")

    #mask all src images
    ret, mask_src = cv2.threshold(cv2.cvtColor(mask_src, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    i = 0
    for img_src in img_srcs:
        img_srcs[i] = cv2.bitwise_and(img_src, img_src, mask=mask_src)
        i+=1

    #sphere normal map
    img_normal = SphereNormalMap(mask_chrome)
    print("Sphere Normal Map")
    
    #sphere specular point
    points = LightSpecularPoint(img_chromes, mask_chrome)
    print("Speclar Points")

    #estimate light source direction
    L_list = LightSourceDirection(points, img_normal)
    print("Light Source Directions")

    img_normal, img_albedo = PMS(img_srcs, L_list)
    print("Finish Photometric Stereo")

    cv2.imwrite("result/normal.png", img_normal)
    cv2.imwrite("result/albedo.png", img_albedo)
    
    cv2.imshow("normal", img_normal)
    cv2.imshow("albedo", img_albedo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    sys.exit(main())

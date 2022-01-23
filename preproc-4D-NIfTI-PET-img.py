# -*- coding: utf-8 -*-

import os
from nilearn.image import load_img, mean_img, index_img, get_data, resample_img
import nilearn.plotting as nip
from nilearn.plotting import plot_img, view_img, show
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math

#%% 任意のスライスのみを抽出するための関数
#   ClairvivoPETで2匹同時撮像を行ったときに解析上の都合から1個体のみの画像を作製する必要がある。

def ExtractSlices(inputDir, inputFile, startS, stopS, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #startS : 抽出する画像の開始スライス（一番手前が0）
    #stopS : 抽出する画像の終了スライス
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')
    #print(niimg.shape)

    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    ImgArryData = niimg.get_fdata()

    # スライスの抽出
    if len(ImgArryData.shape)==3:
        ExtractedArrayNii = ImgArryData[:,:,startS:stopS]   # 3D
    elif len(ImgArryData.shape)==4:
        ExtractedArrayNii = ImgArryData[:,:,startS:stopS,:] # 4D
    else:
        pass
    print(ExtractedArrayNii.shape)
    
    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換
    ConvertImg = nib.Nifti1Image(ExtractedArrayNii, None, header=new_header)
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% 回転・平行移動をするための関数
#   ClairvivoPET/CTのデータはLR軸を中心に‐90度回転させないと正確な軸にならない。
#   また、2匹同時撮像の場合、奥側に配置した個体を手前に配置した個体と同じようにあつかうために回転が必要

def AffineTransform(inputDir, inputFile, TransformMatrix, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #TransformMatrix : affine変換行列（4x4）
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')
    
    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    ImgArryData = niimg.get_fdata()

    # affine変換後の回転成分を計算
    # 変換行列にオリジナルのaffine行列をかける、順番に注意
    newAffineMatrix = np.dot(TransformMatrix, niimg.header.get_best_affine())

    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換
    ConvertImg = nib.Nifti1Image(ImgArryData, affine=newAffineMatrix, header=new_header)
    print('Completed: ' + '/' + inputFile + '.nii')

    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))

# samples transform matrix
# PET手前用
LR_90rotate = np.array([[1, 0, 0, -22],
                        [0, 0, 1, 297],
                        [0, -1, 0, -8],
                        [0, 0, 0, 1]])

# PET奥側用
LR90_IS180rotateBack = np.array([[-1, 0, 0, -20],
                                 [0, 0, -1, -310],
                                 [0, -1, 0, -8],
                                 [0, 0, 0, 1]])

# CT手前用
LR_90rotate_CT = np.array([[1, 0, 0, -20],
                           [0, 0, 1, 300],
                           [0, -1, 0, -8],
                           [0, 0, 0, 1]])

# CT奥側用
LR90_IS180rotate_CT = np.array([[-1, 0, 0, -17],
                                [0, 0, -1, -255],
                                [0, -1, 0, -7],
                                [0, 0, 0, 1]])


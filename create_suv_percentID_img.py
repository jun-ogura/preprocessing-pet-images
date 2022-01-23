# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
from nibabel.processing import smooth_image
from nilearn.image import load_img, mean_img, index_img, resample_img, concat_imgs, iter_img
from nilearn.regions import img_to_signals_labels
import nipype.interfaces.spm as spm
import pandas as pd

#%% create SUV image

def createSUVimg(inputDir, FileName, Weight_g, ID_MBq, outputDir, prefix, suffix):
    # inputDir: 
    # FileName: 拡張子は含めない
    # Weight_g: 
    # ID_MBq: Final Injection Dose (MBq) of tracer
    # outputDir: 出力ディレクトリパス
    # prefix: 出力ファイル名の接頭語
    # suffix: 出力ファイル名の接尾語
    
    # load a non-SUV NIfTI image
    tmpPathName = inputDir + '/' + FileName + '.nii'
    niimg = nib.load(tmpPathName)

    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    LoadImgArryData = niimg.get_fdata()

    # 体重と最終投与量をもとにSUVを計算する
    # MBq -> kBq
    # g -> kg
    CalcImg = LoadImgArryData/((ID_MBq*1000)/(Weight_g/1000))
    
    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, None, header=new_header)
    print('Completed: ' + '/' + FileName + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + FileName + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% 位置合わせ用画像を作製するための関数
# dynamic画像の最初の方のフレームを加算平均して脳もしくは頭部の輪郭がある程度わかるような画像を作製し、
# 他のモダリティ画像との位置合わせに使用する。

def createMeanFramesImg(inputDir, inputFile, frameRnage, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #frameRnage : どこからどこまでのフレームを使って加算平均するかを指定（slice()関数で指定する）
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')
    tmpArrayNiimg = niimg.get_fdata()

    # get single image
    if len(tmpArrayNiimg.shape)==3:
        print('Can not use the 3D image. Use a 4D image only.')
        
    elif len(tmpArrayNiimg.shape)==4:
        multi_frame_img = index_img(niimg, frameRnage) # 4D data (maluti frames)
        single_mean_image = mean_img(multi_frame_img)
        
    else:
        pass
    
    single_mean_imageArray = single_mean_image.get_fdata()

    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換
    ConvertImg = nib.Nifti1Image(single_mean_imageArray, None, header=new_header)
    
    # infomation
    tmpArray = multi_frame_img.get_fdata()
    print('Complete: /' + inputFile + '.nii  ---   ' + 'shape: ' + str(tmpArray.shape))
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% 位置合わせの終わった画像のヘッダー情報をもとに回転・平行移動をするための関数
#   上記のAffineTransform()で出てきた画像の位置合わせを、すでに位置合わせが終わっている画像をもとに
#   行う場合はこの関数を使用する。SPM（Reorient Imageに相当）では変換行列ファイルが必要になるが、この関数はその変換行列を
#   位置合わせ済みの画像のヘッダーから抽出して利用する。

#   ここでは上記のcreateMeanFramesImg()で作製したファイルを位置合わせしたあとに、その位置合わせ済み画像をもとに他のdynamicとstaticの画像の位置合わせを行う

def AffineTransformByAlignedImg(inputDir, inputFile, alignedImgFile, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #alignedImgFile : テンプレート画像に位置合わせが終わっている画像ファイル（パスおよび拡張子含めて指定）
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    '''位置合わせしたい画像の読み込み'''
    niimg = load_img(inputDir + '/' + inputFile + '.nii')
    '''位置合わせ済み画像の読み込み'''
    alignedImgnii = load_img(alignedImgFile)
    
    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    ImgArryData = niimg.get_fdata()

    # 位置合わせ済み画像からaffine変換行列を抽出
    newAffineMatrix = alignedImgnii.header.get_best_affine()

    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換
    ConvertImg = nib.Nifti1Image(ImgArryData, affine=newAffineMatrix, header=new_header)
    print('Completed: ' + '/' + inputFile + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% 画像のCropping処理のための関数
# 下のリンクをもとに作成しているが原点（0,0,0）が画像の中央となる場合にのみ使用可能
# ブレグマなど別の点を原点と指定場合には修正が必要
#
# (参照)Resizing, reshaping and resampling nifti files
# https://www.kaggle.com/mechaman/resizing-reshaping-and-resampling-nifti-files

def createCroppedImg(inputDir, inputFile, target_shape_for_affine, target_shape_for_dim, resolution, interpolation, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #target_shape_for_affine : Cropする大きさ（numpy.array形式、テンプレート画像のqoffset_x,y,zを解像度サイズで除して符号を反転したもの）
    #target_shape_for_dim : Cropする位置（numpy.array形式、テンプレート画像のヘッダーのdim情報＝画像のshapeのこと）
    #resolution : ボクセルサイズ（ここでダウン・アップサンプリングも可能）
    #interpolation: 'continuous' (default), 'linear', or 'nearest'
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')

    # Crop用の4x4変換行列を作製する
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(resolution)
    new_affine[:3,3] = target_shape_for_affine*resolution/2.*-1
    new_affine[3,3] = 1.
    
    # Cropping (resampling)
    Croppednii = resample_img(niimg, 
                              target_affine=new_affine, 
                              target_shape=target_shape_for_dim, 
                              interpolation=interpolation)
    print('Completed: ' + '/' + inputFile + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(Croppednii, os.path.join(outputPathName))


def createCroppedImg_for_MultiFrame(inputDir, inputFile, target_shape_for_affine, target_shape_for_dim, resolution, interpolation, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #target_shape_for_affine : Cropする大きさ（numpy.array形式、テンプレート画像のqoffset_x,y,zを解像度サイズで除して符号を反転したもの）
    #target_shape_for_dim : Cropする位置（numpy.array形式、テンプレート画像のヘッダーのdim情報＝画像のshapeのこと）
    #resolution : ボクセルサイズ（ここでダウン・アップサンプリングも可能）
    #interpolation: 'continuous' (default), 'linear', or 'nearest'
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')

    # Crop用の4x4変換行列を作製する
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(resolution)
    new_affine[:3,3] = target_shape_for_affine*resolution/2.*-1
    new_affine[3,3] = 1.
    
    # Cropping (resampling)
    first_single_img = index_img(niimg, 0)
    Croppednii = resample_img(first_single_img, 
                              target_affine=new_affine, 
                              target_shape=target_shape_for_dim, 
                              interpolation=interpolation)
    '''
    一旦、ダブルフレームにしておく（これはダミーフレームとして使う）
    resample_img()で作ったものは3Dなので、後の処理のことを考えてここでは4Dデータにしておく
    index_img(Croppednii, slice(0, 1))とすることで1フレームの4Dデータとすることができる
    index_img(Croppednii, 0)だと3Dデータになってしまうので注意
    '''
    Croppednii = concat_imgs([Croppednii, Croppednii])
    Croppednii = index_img(Croppednii, slice(0, 1))
    
    for next_single_img in iter_img(niimg):
        next_Croppednii = resample_img(next_single_img, 
                                  target_affine=new_affine, 
                                  target_shape=target_shape_for_dim, 
                                  interpolation=interpolation)
        next_Croppednii = concat_imgs([next_Croppednii, next_Croppednii])
        next_Croppednii = index_img(next_Croppednii, slice(0, 1))
        Croppednii = concat_imgs([Croppednii, next_Croppednii])
    
    # 頭の余計なフレームをそぎ落とす
    Croppednii = index_img(Croppednii, slice(1, Croppednii.shape[3]))
    print('Completed: ' + '/' + inputFile + '.nii,' + '---> saving now')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(Croppednii, os.path.join(outputPathName))


# テンプレート画像のヘッダー情報を見るための関数
    
def showHeaderNIfTI(inputDir, inputFile):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）

    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')
    
    print(niimg.header)


#%% Smoothing処理を行うための関数

def createSmoothedImg(inputDir, inputFile, fwhm, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #fwhm : 単位はmm
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')
    
    # Smoothing
    Smoothednii = smooth_image(niimg, fwhm)
    print('Completed: ' + '/' + inputFile + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(Smoothednii, os.path.join(outputPathName))


#%% create %ID/ml image

# SUV変換前の画像を%ID/ml画像にするときはこの関数を使用（Clairvivo PET専用）
def create_PrecentID_per_ml_img(inputDir, FileName, ID_MBq, outputDir, prefix, suffix):
    # inputDir: 
    # FileName: 拡張子は含めない
    # ID_MBq: Final Injection Dose (MBq) of tracer
    # outputDir: 出力ディレクトリパス
    # prefix: 出力ファイル名の接頭語
    # suffix: 出力ファイル名の接尾語
    
    # load a non-SUV NIfTI image (unit: Bq/ml)
    tmpPathName = inputDir + '/' + FileName + '.nii'
    niimg = nib.load(tmpPathName)

    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    LoadImgArryData = niimg.get_fdata()

    # 最終投与量をもとに %ID/ml を計算する
    # もとの画素値がBq単位なので、最終投与濃度をMBq > Bqに変換して計算
    CalcImg = (LoadImgArryData/(ID_MBq*10**6))*100
    
    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, None, header=new_header)
    print('Completed: ' + '/' + FileName + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + FileName + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% SUV変換済みの画像を%ID/ml画像にするときはこの関数を使用（Clairvivo PET専用）
def create_SUVimg2PrecentID_per_ml_img(inputDir, FileName, Weight_g, ID_MBq, outputDir, prefix, suffix):
    # inputDir: 
    # FileName: 拡張子は含めない
    # Weight_g: 
    # ID_MBq: Final Injection Dose (MBq) of tracer
    # outputDir: 出力ディレクトリパス
    # prefix: 出力ファイル名の接頭語
    # suffix: 出力ファイル名の接尾語
    
    # load a SUV NIfTI image
    tmpPathName = inputDir + '/' + FileName + '.nii'
    niimg = nib.load(tmpPathName)

    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    LoadImgArryData = niimg.get_fdata()

    # SUV画像をもとのBq/mlの画像に戻す
    Bq_pet_ml_Img = LoadImgArryData*((ID_MBq*1000)/(Weight_g/1000))

    # 最終投与量をもとに %ID/ml を計算する
    # もとの画素値がBq単位なので、最終投与濃度をMBq > Bqに変換して計算
    CalcImg = (Bq_pet_ml_Img/(ID_MBq*10**6))*100
    
    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, None, header=new_header)
    print('Completed: ' + '/' + FileName + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + FileName + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))

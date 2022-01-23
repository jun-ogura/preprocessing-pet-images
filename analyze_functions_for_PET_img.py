# -*- coding: utf-8 -*-

import os
from nilearn.image import load_img, mean_img, index_img, resample_img, concat_imgs, iter_img
import nibabel as nib
import numpy as np
from nibabel.processing import smooth_image
from nilearn.regions import img_to_signals_labels
import pandas as pd

#%% マルチフレームデータ（4D NIfTI）から任意のフレームを抽出し、その平均画像を作製する関数
def CreateMeanImgToMultiFrm(inputDir, inputFile, startF, stopF, outputDir, prefix, suffix):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFile : 入力ファイル（4D NIfTI形式）
    #startF : 平均画像作製に使用する画像の開始フレーム
    #stopF : 平均画像作製に使用する画像の終了フレーム
    #outputDir : 出力ディレクトリの指定
    #prefix : 出力ファイル名の接頭語
    #suffix : 出力ファイル名の接尾語
    
    # import 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFile + '.nii')
    #print(niimg.shape)
    
    # get multi frames
    multi_mni_image = index_img(niimg, slice(startF-1, stopF))
    print(multi_mni_image.shape)
    
    # create mean image from multi frames
    mean_mni_image = mean_img(multi_mni_image)
    
    # seve mean image
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(mean_mni_image, os.path.join(outputPathName))


#%% VOI解析をするための関数
# 単一ファイル解析用
def exportVOIvalue(inputDir, FileName, label_img, label_txt, outputDir, prefix, suffix, outputFlag):
    # inputDir: 
    # FileName: 拡張子は含めない
    # outputDir: 出力ディレクトリパス
    # label_img: VOI template image (.nii)
    # label_txt: tab区切りのテキストファイル（SAMIT toolboxのラベルファイルを参照）
    # prefix: 出力ファイル名の接頭語
    # suffix: 出力ファイル名の接尾語
    # outputFlag: 0 -> ファイル出力, 1 -> オブジェクトをreturn
    
    # load NIfTI image
    tmpPathName = inputDir + '/' + FileName + '.nii'
    niimg = nib.load(tmpPathName)

    # img_to_signals_labels()が同じshapeかつ同じaffineでないと処理してくれないので修正する
    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    ImgArryData = niimg.get_fdata()

    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # load a label NIfTI image
    tmpPathName = inputDir + '/' + label_img + '.nii'
    labelimg = nib.load(tmpPathName)

    # NumpyArrayからNIfTI形式に変換
    ConvertImg = nib.Nifti1Image(ImgArryData, affine=labelimg.header.get_best_affine(), header=new_header)
        
    if len(niimg.shape)==3:
        # シングルフレームの場合
        # img_to_signals_labels()が4Dデータしか扱えないので一時的にダミー4Dにする
        niimg = concat_imgs([ConvertImg, ConvertImg])
        Flag_M_Frames = 0
    else:
        # マルチフレームの場合
        # なぜかこの処理をしないとaffineが完全一致しているにも関わらず、一致していないと言われてしまうので仕方なく実行
        niimg = concat_imgs([ConvertImg])
        Flag_M_Frames = 1

    
    # import VOI label
    tmpPathName = inputDir + '/' + label_txt + '.txt'
    label_pd = pd.read_csv(tmpPathName, sep='\t',header=None, skiprows=3)
    label_pd.columns = ['index', 'name']

    # result VOI
    # img_to_signals_labels()で得られるsignalsはnumpy.ndarray形式なのでflatten()を使って平坦化する
    resVOI = img_to_signals_labels(niimg, labelimg, mask_img=None, background_label=0, order='F', strategy='mean')
    resVOI = resVOI[0].flatten()
   
    # ファイル名pd.series
    frames = [0] # initialize
    for i in range(1, niimg.shape[3]+1):
        tmp_frames = [i]*len(label_pd)
        frames.extend(tmp_frames)
    frames = frames[1:]
    
    if Flag_M_Frames==0:
        resVOI = resVOI[:len(label_pd)] # ダブルフレームになっているのでシングルに修正
        #IDs = IDs[:len(label_pd)]
        frames = frames[:len(label_pd)]
        FileName_pd = pd.DataFrame({'File name': [FileName]*len(label_pd),
                                    #'File ID': IDs,
                                    'Frame No': frames})
    else:
        FileName_pd = pd.DataFrame({'File name': [FileName]*len(label_pd)*niimg.shape[3],
                                    #'File ID': IDs,
                                    'Frame No': frames})
        label_pd = pd.concat([label_pd]*niimg.shape[3], axis=0).reset_index(drop=True) # フレーム分だけ繰り返したらインデックスを修正
    
    # result VOI最終版
    resVOI_pd = pd.DataFrame({'mean': resVOI})

    # VOI解析結果の整形
    outputResVOI = pd.concat([FileName_pd, label_pd, resVOI_pd], axis=1)
    
    if outputFlag==0:
        # テキストとして出力
        outputPathName = outputDir + '/' + prefix + FileName + suffix + '.csv'
        outputResVOI.to_csv(outputPathName, header=True, index=False, sep=',')
        
    elif outputFlag==1:
        # オブジェクトとして出力
        return(outputResVOI)

#%% VOI解析をするための関数
# 複数ファイル解析用
def exportVOIvalues(inputDir, inputFileArray, label_img, label_txt, outputDir, outputFile, prefix, suffix):
    # inputDir: 
    #inputFileArray : 入力ファイル名（4D NIfTI形式）をnumpy.Array形式で与える
    # outputDir: 出力ディレクトリパス
    # outputFile: 出力ファイル名
    # prefix: 出力ファイル名の接頭語
    # suffix: 出力ファイル名の接尾語

    # 結果格納用の空のpandasデータフレームを作製
    VOItbl = pd.DataFrame({'File name': [],
                           'File ID': [],
                           'Frame No': [],
                           'index': [],
                           'name': [],
                           'mean': []})
    
    # ファイルごとにVOI解析し、結果をVOItblに追加していく
    for FileName in range(0, len(inputFileArray)):
        tmpTbl = exportVOIvalue(inputDir, inputFileArray[FileName], label_img, label_txt, '', '', '', 1)
        #IDs = [FileName]*len(label_pd)*niimg.shape[3]
        VOItbl = pd.concat([VOItbl, tmpTbl], axis=0)
        print('Processed: ' + inputFileArray[FileName] + '.nii')

    # テキストとして出力
    outputPathName = outputDir + '/' + prefix + outputFile + suffix + '.csv'
    VOItbl.to_csv(outputPathName, header=True, index=False, sep=',')


#%% SUVr（リファレンスVOIで標準化した）画像を作製するための関数
def createSUVrImg_PMOD(inputDir, FileName, refVOIval, outputDir, prefix, suffix):
    # inputDir: 
    # FileName: SUV画像を指定する。拡張子は含めない
    # refVOIval: 入力SUV画像のリファレンスにしたいVOIの値
    # outputDir: 出力ディレクトリパス
    # prefix: 出力ファイル名の接頭語
    # suffix: 出力ファイル名の接尾語
    
    # load a non-SUV NIfTI image
    tmpPathName = inputDir + '/' + FileName + '.nii'
    niimg = nib.load(tmpPathName)

    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    LoadImgArryData = niimg.get_fdata()

    # SUVrを計算する（PMODの計算式）
    CalcImg = LoadImgArryData/refVOIval
    
    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, None, header=new_header)
    print('Completed: ' + '/' + FileName + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + FileName + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


def createSUVrImg_SAMIT(inputDir, FileName, refVOIval, Weight_g, ID_MBq, outputDir, prefix, suffix):
    # inputDir: 
    # FileName: SUV画像を指定する。拡張子は含めない
    # refVOIval: 入力SUV画像のリファレンスにしたいVOIの値
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

    # 体重と最終投与量をもとにSUVrを補正計算する（SAMIT toolboxの計算式）
    # MBq -> kBq
    # g -> kg
    CalcImg = (LoadImgArryData/refVOIval)/((ID_MBq*1000)/(Weight_g/1000))
    
    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, None, header=new_header)
    print('Completed: ' + '/' + FileName + '.nii')
    
    # .niiとして出力
    outputPathName = outputDir + '/' + prefix + FileName + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% 平均画像を作製するための関数
# 複数個体の画像の平均画像を作製
def createMeanImg(inputDir, inputFileArray, outputDir, outputFile):
    #inputDir : 入力ファイルのあるディレクトリ
    #inputFileArray : 入力ファイル名（4D NIfTI形式）をnumpy.Array形式で与える
    #outputDir : 出力ディレクトリの指定
    #outputFile : 出力ファイル名
    
    # num of input file
    numInputFlile = inputFileArray.shape[0]
    
    # first import of 4D NIfTI
    niimg = load_img(inputDir + '/' + inputFileArray[0] + '.nii')
    print('Processed: ' + inputFileArray[0] + '.nii')
    
    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    ImgArryData = niimg.get_fdata()
    
    # next import of 4D NIfTI (loop)
    for inputFile in range(1, numInputFlile):
        nextNiimg = load_img(inputDir + '/' + inputFileArray[inputFile] + '.nii')        
        nextImgArryData = nextNiimg.get_fdata()

        # 加算
        ImgArryData = ImgArryData + nextImgArryData
        print('Processed: ' + inputFileArray[inputFile] + '.nii')
    
    # 平均
    MeanImg = ImgArryData/numInputFlile
    print('Total ' + str(numInputFlile) + ' images')
        
    # 読み込みデータからヘッダーのコピーを作製
    new_header = niimg.header.copy()

    # NumpyArrayからNIfTI形式に変換
    ConvertImg = nib.Nifti1Image(MeanImg, affine=niimg.header.get_best_affine(), header=new_header)

    # seve mean image
    # .niiとして出力
    outputPathName = outputDir + '/' + outputFile + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% 差分画像の作製
# 2つの画像を読み込む（2つの画像は同じディレクトリに置いておくこと）
def CalcDifferenceImg(inputDir, InputFileNameA, InputFileNameB, outputDir, outputFileName):
    # inputDir : 入力画像のパス
    # InputFileNameA : 入力画像のファイル名（引かれる側）
    # InputFileNameB : 入力画像のファイル名（引く側）
    #      →　A - B　となる
    # outputDir : 差分画像の出力先パス
    # outputFileName : 差分画像のファイル名
    
    tmpPathNameA = inputDir + InputFileNameA
    tmpPathNameB = inputDir + InputFileNameB
    
    tmpLoadImgA = nib.load(tmpPathNameA)
    tmpLoadImgB = nib.load(tmpPathNameB)
    
    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    LoadImgArryDataA = tmpLoadImgA.get_fdata()
    LoadImgArryDataB = tmpLoadImgB.get_fdata()
    
    # 2つの画像の差分を計算する
    CalcImg = LoadImgArryDataA - LoadImgArryDataB
    
    # 読み込んだ画像からアフィン変換に必要な行列を取得
    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, affine=tmpLoadImgA.header.get_best_affine())
    
    # .niiとして出力
    outputPathName = outputDir + outputFileName + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% 割合（百分率）画像の作製
# 2つの画像を読み込む（2つの画像は同じディレクトリに置いておくこと）
def CalcRatioImg(inputDir, InputFileNameA, InputFileNameB, outputDir, outputFileName):
    # inputDir : 入力画像のパス
    # InputFileNameA : 入力画像のファイル名（分子となる画像）
    # InputFileNameB : 入力画像のファイル名（分母となる画像）
    #      →　A / B　となる
    # outputDir : 割合画像の出力先パス
    # outputFileName : 割合画像のファイル名
    
    tmpPathNameA = inputDir + InputFileNameA
    tmpPathNameB = inputDir + InputFileNameB
    
    tmpLoadImgA = nib.load(tmpPathNameA)
    tmpLoadImgB = nib.load(tmpPathNameB)
    
    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    LoadImgArryDataA = tmpLoadImgA.get_fdata()
    LoadImgArryDataB = tmpLoadImgB.get_fdata()
    
    # 画像Bに対する画像Aの割合を計算する
    CalcImg = (LoadImgArryDataA/LoadImgArryDataB)*100
    
    # 読み込んだ画像からアフィン変換に必要な行列を取得
    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, affine=tmpLoadImgA.header.get_best_affine())
    
    # .niiとして出力
    outputPathName = outputDir + outputFileName + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


#%% マスク適用画像の作製
# マスク画像とマスクをかけたい画像を読み込む
def CalcMaskImg(MaskFilePathName, inputDir, inputFile, outputDir, prefix, suffix):
    # MaskFilePathName : マスク画像のパス名含むファイル名（フルパス）
    # inputDir : マスク適用画像のあるディレクトリ
    # inputFile : マスク適用画像のファイル名
    # outputDir : マスク適用画像の出力ディレクトリ
    #　prefix : 出力ファイル名の接頭語
    #　suffix : 出力ファイル名の接尾語

    tmpLoadImgMask = nib.load(MaskFilePathName)
    tmpLoadImg = nib.load(inputDir + '/' + inputFile + '.nii')
    
    # 読み込んだ画像はNiBabel形式なのでNumpy.Array形式で扱えるようにする
    LoadImgArryDataMask = tmpLoadImgMask.get_fdata()
    LoadImgArryData = tmpLoadImg.get_fdata()
    
    # マスク画像とマスクを適用したい画像のshapeが一致していないと処理ができない
    # 一致していない場合は以下の1行を編集して使用すること
    # np.delete()の第2引数は削除したいスライス位置、第3引数は0=行、1=列、2=行列の削除を表す
    # 通常はこの1行はコメントアウトしておくこと
    LoadImgArryDataMask = np.delete(LoadImgArryDataMask, 96, 2) # 今回はマスク以外のテンプレートが(96,120,96)なのでそれに合わせた
    
    # 上記のイレギュラー処理が必要な場合はコンソールに警告を出す
    if (tmpLoadImg.shape == tmpLoadImgMask.shape):
        pass
    else:
        print('******************************************************')
        print('：：：警告：：：')
        print('')
        print('マスク画像とマスクを適用したい画像のshapeが一致していません！')
        print('関数内のnp.delete()を適切に編集してshapeが同じになるようにしてください。')
        print('******************************************************')
        print('')

    # 画像Bに対する画像Aの割合を計算する（マスクを欠けた部分を0にする）
    CalcImg = LoadImgArryData*LoadImgArryDataMask
    
    # 読み込みデータからヘッダーのコピーを作製
    new_header = tmpLoadImg.header.copy()

    # 読み込んだ画像からアフィン変換に必要な行列を取得
    # NumpyArrayからNIfTI形式に変換する
    ConvertImg = nib.Nifti1Image(CalcImg, affine=tmpLoadImg.header.get_best_affine(), header=new_header)
    
    print('Completed: ' + '/' + inputFile + '.nii')

    # .niiとして出力
    outputDir + '/' + prefix + inputFile + suffix + '.nii'
    outputPathName = outputDir + '/' + prefix + inputFile + suffix + '.nii'
    nib.save(ConvertImg, os.path.join(outputPathName))


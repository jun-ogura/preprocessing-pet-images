# preprocessing-pet-images
These are pet image preprocessing functions using python.

小動物用PET/CTイメージングシステムであるClairvivoPET/CT（島津製作所）で得た画像をPythonで処理するための関数です。

暫定版であるため、随時、修正を行います。

使用するための必要な追加パッケージは以下です：

- numpy
- pandas
- nibabel
- nilearn

処理したい画像はdcm2nii等を使って4D NIfTI nii (.nii)に変換したファイルを使用します。

ここに掲載している関数はnibabelやnilearnに含まれる各種機能のうちPET画像処理で必要になってくる機能をピックアップして使いやすく関数化したものになります。

[nibabel ](https://nipy.org/nibabel/)や[ nilearn ](https://nilearn.github.io/stable/index.html)の詳細については公式Webサイトをご参照ください。

なお、ClairvivoPET/CTは厳密には一体型PET/CTではなく、PETとCTという独立したシステムとなります。

ある程度PETとCT画像の位置は合うように設定されているはずですが、なぜか合わないことの方が多いです。

そのため、回転・平行移動処理ができる関数も含めています。

細かい位置修正は[ 3D Slicer ](https://www.slicer.org/)などの他のソフトウェアをご利用ください。


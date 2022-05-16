# Pet Image Processing Functions using Python.

小動物用PET/CTイメージングシステムであるClairvivoPET/CT（島津製作所）で得た画像をPythonで処理するための関数です。<br>
nibabelとnilearnのうち、PET/CT画像の処理に役立つメソッドをユーザーフレンドリーに使いやすく関数化したものです。<br>

メンテナンス等は小倉が行っており、自分の研究目的で作成したものです。<br>
暫定版であるため、随時、修正を行います。<br>

リクエストがあれば Pull request をお願い致します。<br>
ここではPETデータ解析を完全無料で行うことを目的とし、核医学研究への貢献を目指します。

ソースコードはjupyter notebook形式で公開しており、google colaboratoryのリンクも付けてあります。<br>
公開ファイルは現在、以下の2つとなっております（GitHub Pageからこのサイトに訪問された方は右上にある "Veiw on GitHub"ボタンのリンクからアクセスください）。

1. animal_PET_img_processing.ipynb
2. animal_PET_img_other_functions.ipynb

1は最低限必要と思われる処理を行うための関数を収録しています。

2は状況に応じて必要になると思われる処理を行うための関数を収録しています。

これらの関数を使用するために必要な追加パッケージは以下となります：

- numpy
- pandas
- nibabel
- nilearn

処理したい画像はdcm2nii等を使って4D NIfTI nii (.nii)に変換したファイルを使用します。

ここに掲載している関数はnibabelやnilearnに含まれる各種機能のうちPET画像処理で必要になってくる機能をピックアップして使いやすく関数化したものになります。

[nibabel ](https://nipy.org/nibabel/)や[ nilearn ](https://nilearn.github.io/stable/index.html)の詳細については公式Webサイトをご参照ください。

なお、ClairvivoPET/CTは厳密には一体型PET/CTではなく、PETとCTという独立したシステムとなります。<br>
ある程度PETとCT画像の位置は合うように設定されているはずですが、なぜか合わないことが多いです。<br>
そのため、回転・平行移動処理ができる関数も含めています。<br>

もちろんClairvivoPET/CT以外の機種で撮像したデータも処理可能です。

細かい位置修正は[ 3D Slicer ](https://www.slicer.org/)などの他のソフトウェアをご利用ください。


---

※　本ソースコードの使用にて生じた損害などに関しては一切の責任を負いかねますのであらかじめご了承ください。
<br>
<br>
<br>

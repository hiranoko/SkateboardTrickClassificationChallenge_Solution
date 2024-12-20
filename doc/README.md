# スケートボードトリック分類チャレンジの解法

Solution for Motion Decoding Using Biosignals

Author : Kot

Score : 

```
データ１ 0.9155496  (1st-place)
データ２ 0.7104558  (6th-place)
```

## Abstract

　本稿では、脳波や筋電位などの生体信号を用いてスケートボードのトリックを分類するニューラルネットワークを提案します。提案モデルは、畳み込みとプーリングを組み合わせた基本的なエンコーダーです。モデルの構造は、ベイズ最適化を用いたハイパーパラメータ探索により決定しました。その結果、生信号データ（データ①）では高周波数帯域の特徴が、クレンジング処理後のデータ（データ②）では低周波数帯域の特徴が分類に重要であることが判明しました。本手法により、データ1で0.9008、データ2で0.6702のスコアを達成しました。コードは[URL](https://github.com/hiranoko/SignateBiosignal)よりダウンロードできます。

## Introduction

　本コンテストは、スケートボード上での動作を頭皮上の生体信号（国際10-10法）からポンピング、前向きキックターン、後向きキックターンの3つのトリックを分類することを目的としています。データは、生の生体信号（データ①）と生体信号に由来しない成分をクレンジング処理したもの（データ②）の2種類を用いて分析を行います。
頭皮上から取得される生体信号データは、動作中で多くのノイズを含むため適切な前処理が重要です。この実環境でのスケートボーダーの動作予測は、ブレイン・マシンインターフェースやヒューマノイドロボットの運動生成研究への応用が期待されます。スケートボードトリックの分類を通じて、最適なデータ前処理とモデル構築手法を探求しました。

## Dataset

　コンテストのデータセットは、被験者（subject）ごとに取得したファイルが提供されました。国際10-10法で決められた72チャンネルの電気信号、トリックの時刻と種別、チャンネルラベルなどがファイルに格納されています。Sampling Rateは500Hzで、訓練データは全区間の信号、推論データは0.5秒間に切り出された形式で配布されました。

<div style="text-align: center;">
  <h3>Table 被験者一覧</h3>
</div>

| id       | foot  | hand  |
| -------- | ----- | ----- |
| subject0 | left  | Right |
| subject1 | right | Right |
| subject2 | right | Right |
| subject3 | right | Left  |
| subject4 | left  | Right |

### データ①

最低限の前処理のみ施されたもの。

- 取得失敗チャネルデータの0埋め

### データ②

先行研究に基づくパイプライン(Callan, et al., 2024)が施されたもの。

- バンドパスフィルタ（3～100Hz）
- 電源ノイズ（60Hz）除去
- 駆動していないチャンネルを除去
- アーチファクト部分空間法（ASR）
- 部分的に駆動していないチャンネル信号を補完
- レファレンス信号の平均化
情報最大化基準独立成分分析による低次元化
- ダイポール推定法による低次元化信号からの脳波再構成
- ICLabel法による脳波成分分解
- 非「脳由来」成分（眼電、筋電、など）の除去（閾値50%）
- 非「脳由来」成分が全て除かれているわけではないことに注意。

## データ処理の工夫点と考察

　本節では、データの前処理と拡張の検証内容をまとめます。検証時は1次元畳み込み (Conv1D) に基づくU-Netアーキテクチャを使用しました。また、モデリングにあたって探索的データ解析で得られた知見についてまとめます。

### 前処理

　データ①の生体信号にはノイズが含まれるため、適切なフィルタリングが必要です。フィルターのカットオフ周波数は、観測可能な周波数帯域とナイキスト周波数から以下のように設定しました。

- **ハイパスフィルター**：2Hz（0.5秒間で観測可能な最低周波数）
- **ローパスフィルター**：125Hz（ナイキスト周波数の1/2）

この帯域は、脳波（EEG）と筋電位（EMG）を含む周波数帯です。また、電源ノイズ（50/60Hz）の影響を考慮し、ノッチフィルターも試しました。フィルタリングの有無によるモデルの性能を、測定ファイル単位の交差検証（k=3）で確認しました。結果を表に示します。

<div style="text-align: center;">
  <h3>Table 前処理の比較（データ①）</h3>
</div>

| Experiment                | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 | Experiment 5 |
| :------------------------ | :----------: | :----------: | :----------: | :----------: | :----------: |
| Normalize                 |      ✓       |      ✓       |      ✓       |      ✓       |      ✓       |
| Highpass Filter (2Hz)     |              |      ✓       |              |              |              |
| Lowpass Filter (125Hz)    |              |              |      ✓       |              |              |
| Bundpass Filter (2~125Hz) |              |              |              |      ✓       |      ✓       |
| Notch Filter (60Hz)       |              |              |              |              |      ✓       |
| OOF(Kfold)                |    0.788     |    0.811     |    0.788     |  **0.847**   |    0.832     |

表からバンドパスフィルターを使用することで、精度は向上しているのがわかります。一方、ノッチフィルターによる電源ノイズの除去は精度がみられません。したがって、生体信号に関する情報を最大限に活用しつつ、情報の欠損を防ぐために、前処理は最低限にしてバンドパスフィルターのみを採用しました。

### データ拡張

　ニューラルネットワークでは新しいデータに対する汎化性能が重要となります。電極配置の対称性を活用した拡張や信号の符号や振幅、位相などを変えるデータ拡張を検討しました。

1. Cutout
2.	Time Shift
3.	Amplitude Scaling
4.	Add Noise
5.	Roll
6.	Random Channel Flip
7.	Random Sign Flip
8.	Random Channel Masking

これらの**データ拡張はモデルの精度向上に寄与しません**でした。特に、信号の振幅や符号をランダムに変更する手法は、モデル性能を低下させることがわかりました。

### 探索的データ解析

　モデリング前にデータセットについて以下の3点を確認します。

1. 訓練と評価のドメインが同一であるのか
2. 被験者ごとに依存性があるのか
3. 重要なチャンネルはあるのか

#### 1. 訓練と評価のドメインが同一であるのか

　はじめに訓練データであるのか推定するモデルを構築しました。訓練と評価用のデータ数が同一となるようにして学習しました。結果を図に示します。青線はsoftmax後の値、緑線はラベルを表します。

![データ①](https://firebasestorage.googleapis.com:443/v0/b/type-c1c71.appspot.com/o/V5C3QuJFU8Od1GJ2D0uP76gYBdD2%2FvbwUQIWjSp5MPKKg.jpg?alt=media&token=faef295d-53f7-4a26-821e-9f293a44a489)

<div style="text-align: center;">
  <h3>Figure. Adversarial Validation(データ①)</h3>
</div>

![データ②](https://firebasestorage.googleapis.com:443/v0/b/type-c1c71.appspot.com/o/V5C3QuJFU8Od1GJ2D0uP76gYBdD2%2FLfzsjJb27oeomPzy.jpg?alt=media&token=180a39c5-5499-4f95-93ff-05e4534533f7)

<div style="text-align: center;">
  <h3>Figure. Adversarial Validation(データ②)</h3>
</div>

訓練データは正答率100%近くとなり、検証データは50%程度の正答率であることが確認できました。これは全ての信号を記憶できる表現力を有することを表します。検証データに対するsoftmax後の出力は大きな分散が見られ、訓練データを記憶しているために新規データに対して有意な特徴を得てない事からランダムな回答となり、正答率は50%となりました。以上より、**訓練用データと評価用データのドメインは同一である**ことがわかりました。また、**モデルは過学習しないよう単純にすべき**指針を得ました。

#### ②被験者ごとに依存性があるのか

　つぎに被験者を推定するモデルを構築します。図に正解のラベルと推定結果のラベル値を示します。

![データ①](https://firebasestorage.googleapis.com:443/v0/b/type-c1c71.appspot.com/o/V5C3QuJFU8Od1GJ2D0uP76gYBdD2%2FGksL9tkL6TjRbsVP.jpg?alt=media&token=205f2047-99b1-42ed-9f09-4cb93dd4c5f2)

<div style="text-align: center;">
  <h3>Figure </h3>
</div>

![データ②](https://firebasestorage.googleapis.com:443/v0/b/type-c1c71.appspot.com/o/V5C3QuJFU8Od1GJ2D0uP76gYBdD2%2Fr79EzOpjenPWRvRi.jpg?alt=media&token=6a0f9951-8496-4abc-9236-dbf028cef2b6)

<div style="text-align: center;">
  <h3>Figure </h3>
</div>

被験者ごとの交差検証で正答率は100%近くとなり、信号データには被験者依存性があることがわかりました。これは国際10-10法と**標準化された電極配置であっても属人性がある**事を表します。本コンテストの被験者は利き手や利き足に違いはありましたが、生体信号にはそれ以上の被験者依存性が強く現れました。そのため、**被験者ごとにモデルを構築することが重要**になります。  

　学習に使えるデータ数は減少しますが被験者ごとのデータでモデリングする方が正答率は向上します。もし、正答率が100%でない場合は被験者の利き手やスタンスなどを考慮してモデリングすることでデータ数を増やす事や外部データセットの利用が必要だと考えていました。本来は被験者を推定できない方が望ましく、測定データを増やすことでモデルをスケールさせることが必要です。

### ③重要なチャンネルはあるのか？

　最後にUnetのダウンサンプル前にSqueeze-and-Excitationブロックを挿入して学習を行い、チャンネルごとの重要度について確認しました。基本的な信号からの特徴抽出は畳み込みと活性化レイヤーを組み合わせたものです。ベースラインでは畳み込みは72chの全てを使います。但し、入力信号には無効なチャンネルも含まれるため0埋めされたものが含まれます。そこで、分類に寄与している信号には偏りがあると考えてはじめにSE blockを検討しました。チャンネル方向にSE blockを挿入してどのチャンネルに重み付けされるのか調べました。学習の結果を図に示します。横軸はチャンネルインデックス、縦軸は活性度を表す。

![V5C3QuJFU8Od1GJ2D0uP76gYBdD2/i9CwWDUxAnKDy4FD.jpg](https://firebasestorage.googleapis.com:443/v0/b/type-c1c71.appspot.com/o/V5C3QuJFU8Od1GJ2D0uP76gYBdD2%2Fi9CwWDUxAnKDy4FD.jpg?alt=media&token=bce2a87a-b514-4dc3-be26-d81bf620c079)

図の活性度は0.5付近に集中しています。これは特定のチャンネルに依存せず学習していることを示します。アテンションは使用したモデルでは有効に働いていません。一次元の信号分類では浅いレイヤーで複雑な特徴を調べる必要はなく、単純なモデルリングが有効に働きます。また、本コンテストの入力は250サンプルと時系列にも短いため、時系列方向へのアテンションも同様な結果となりました。

## モデリングの工夫点と考察

　本節では、提案モデルをFeature Extractor、Pooling、およびHeadの3つからコンポーネントごとに検証した内容を説明します。1次元畳み込み (Conv1D) に基づくU-Netアーキテクチャをベースラインモデルとして最適なモデル構成を探索しました。

### Feature Extractor

特徴抽出層として、以下の異なるアプローチを試行しました:

- Conv1D: 標準的な畳み込み演算
- Depth-wise: 
- Separable:

　畳み込みで調整可能なパラメータはカーネルサイズ、パディング、ストライドなどがあります。最も重要なものはカーネルサイズであり受容野に相当します。例えば低周波成分が重要な特徴であるのに小さいカーネルサイズで浅いレイヤーだと受容野は不足します。パディングは信号の端部のアーチファクトが効く場合に重要となりますが、提供データは信号は中央部が切り出されて無視できるとして0埋めとしています。ストライドは1と固定して、全区間の信号を活用します。カーネルサイズのみを変えながらどの方法が有効であるのか検証します。

### Pooling

プーリング層は、特徴マップのダウンサンプリングを行うために使用され、次の手法を比較しました:

- Generalized Mean Pooling (GeM): 入力の特徴の幾何平均を用いることで、より柔軟な特徴抽出を可能にする。
- MaxPooling: 最大値を取得することで、最も強い特徴を保持。
- AveragePooling: 平均値を取得し、全体的な傾向を反映させる。

前段のFeature Extractではストライド1で畳み込むため、隣接する特徴は似通っているものが多く、取捨選択が必要となります。
プーリングの有無による精度の比較を示す。

<div style="text-align: center;">
  <h3>Table 前処理の比較（データ②）</h3>
</div>

表よりプーリングを含めると精度が向上していることがわかります。どのプーリングを選択、併用するのか後段で探索します。

### Head

出力部分には以下のモジュールを試しました:

- Convolution + Fully Connected Layers: 畳み込み層を用いて特徴を集約し、全結合層により最終出力を得る。

分類部では、全結合層（Fully Connected Layer）を使用しました。

### ネットワーク構造の探索

ハイパーパラメータの最適化には、Optunaを用いたベイズ最適化を行いました。探索したハイパーパラメータは以下の通りです。

- **隠れ層のチャンネル数**：128〜1024（128刻み）
- **カーネルサイズ**：3〜19（2刻み）
- **畳み込み層の数**：2〜3
- **畳み込みの種類**：Conv1d、Depthwise、Separable
- **プーリングの種類**：Max、Average、Both
- **活性化関数**：ReLU、SiLU、GELU、Leaky ReLU
- **グローバルアベレージプーリングの使用**：True、False

最終的に、以下の設定が最適であることがわかりました。

データ①

| Parameter       | Value      |
| --------------- | ---------- |
| in_channels     | 72         |
| num_classes     | 3          |
| hidden_channels | 768        |
| kernel_size     | 7          |
| stride          | 1          |
| conv_type       | separable  |
| pooling_type    | both       |
| activation      | leaky_relu |
| num_layers      | 2          |
| use_gap         | False      |

データ②

| Parameter       | Value      |
| --------------- | ---------- |
| in_channels     | 72         |
| num_classes     | 3          |
| hidden_channels | 256        |
| kernel_size     | 15         |
| stride          | 1          |
| conv_type       | standard   |
| pooling_type    | avg        |
| activation      | leaky_relu |
| num_layers      | 2          |
| use_gap         | False      |

### Ideas tried but not worked obviously

1. 補助タスク

LED側かLaser側の傾斜
直前のトリック

2. EMA

3. 時系列

TransformerやRNN

## 分析結果から得られたインサイトと考察

1.	被験者依存性の強さ  
被験者ごとのEEG信号の違いが顕著で、個別のモデルを作成することが精度向上に重要であることが確認された。

2.	アーチファクトを含むデータの方が精度が高い  
クレンジング処理されたデータよりも、アーチファクトが含まれる生のEEGデータの方が高精度であり、ノイズも有効な特徴として捉えられている可能性がある。

3.	カーネルサイズの調整とデータの種別依存性  
生データ（データ①）では大きなカーネルサイズを使用することで低周波数成分を効果的に捉え、クレンジングデータ（データ②）では小さなカーネルサイズが有効であった。これは、データの種別に応じて異なる周波数帯域の特徴を抽出するため、カーネルサイズの調整が必要であることを示している。

## データ①②で作成したモデルの分類性能の違いに関する考察

1.	データ精度：データ① > データ②  

生のEEGデータ（データ①）の方が、クレンジングされたデータ（データ②）よりも高い分類精度を示しました。クレンジングによるノイズ除去が、実際には有効な特徴（特に被験者固有の信号パターン）を除去している可能性が考えられます。これにより、ノイズやアーチファクトを除去するよりも、被験者依存性を活用することがモデルの精度向上に貢献していることが示唆されます。

2.	GradCAMによる可視化結果　　

GradCAMを使用して、モデルが波形全体に対してどのように反応しているかを可視化したところ、生データ（データ①）では波形全体にわたって強い反応が確認されました。これにより、モデルが特定の時間区間に依存するのではなく、全体的な信号から有効な特徴を学習していることがわかります。

## 分析結果の社会課題への応用・展開に関する考察

1.	リアルタイム処理への適用可能. 

最適化されたモデルは計算量が比較的軽量であり、リアルタイムでの動作が可能です。特に、FPGAなどの低遅延なハードウェアと組み合わせることで、センサーから分類までのレイテンシーを最小限に抑えることができ、リアルタイムの応答が求められるアプリケーションへの適用が期待されます。

2.	精度の改善余地と用途の検討. 

現時点での分類精度は特定のタスクには不十分ですが、用途に応じて使い道があります。例えば、精度の向上が必須でない状況や、複数のセンサーからのデータを組み合わせて使うことで、精度を補完する形での利用が可能です。さらに、精度改善に向けて、今後はモデルのさらなる最適化や追加のデータ収集が課題となります。

3.	後段処理やハードウェアとの連携. 

分類結果は、リアルタイムでのアクション実行や後段のデータ処理に利用可能です。たとえば、分類された動作データを基にした自動制御システムの作成や、IoTデバイスとの連携を通じた環境適応型インターフェースの構築が考えられます。また、分類結果を蓄積し、将来的な分析や学習モデルの改善に役立てることで、さらなる精度向上や新たな応用が期待できます。

## 生体信号データの取得における課題に対するAIによる解決策に関する考察

1. ノイズやアーチファクトの除去. 

    • 課題:  
生体信号は、外部の電磁的ノイズや筋電図（EMG）、眼電図（EOG）などのアーチファクトによって汚染されやすいです。特に、運動中や日常生活の環境でのデータ取得では、この影響が顕著です。

	•	AIによる解決策:   
AIを活用した自動アーチファクト除去アルゴリズムを開発することで、リアルタイムでノイズを検知し、除去できます。例えば、ディープラーニングを用いた自動フィルタリングや、異常値を検出する異常検知モデルによって、生の生体信号データをリアルタイムでクレンジングする技術が考えられます。これにより、データの品質を向上させ、後続の解析を正確に行うことができます。

2. 長時間のデータ収集とリアルタイム処理. 

	•	課題: 長時間にわたって安定して生体信号を取得することは難しく、特に持続的なモニタリングではデータの欠落やセンサーのズレが発生することがあります。また、リアルタイムでのデータ処理も課題です。
	•	AIによる解決策: AIを用いたデータ補完技術により、欠損したデータやノイズによる異常な信号を予測して補正することが可能です。また、AIベースのリアルタイム処理アルゴリズムをFPGAやエッジデバイスに実装することで、リアルタイムでの生体信号のモニタリングとフィードバックが実現できます。特に、リアルタイムでの異常検知と警告を行うことにより、迅速な対応が可能になります。

3. センサーの装着性と使いやすさ. 

	•	課題: 生体信号を正確に取得するためには、センサーの正しい装着が必要ですが、装着時の違和感や長時間の使用による不快感が問題になることがあります。また、誤った装着によって正確なデータが取得できないことも課題です。
	•	AIによる解決策: AIを用いた装着状態の自動検知や補正機能により、センサーの適切な位置や密着度をリアルタイムでモニタリングできます。装着状態の問題が発生した場合、アラートを送って適切な調整を促すシステムを構築することで、データの正確性を確保できます。さらに、AI技術を活用して装着の影響を最小限に抑えた装置の設計にも貢献できます。

4. 多様なデータの統合と解釈の難しさ. 

	•	課題: 生体信号データは、脳波（EEG）、心電図（ECG）、筋電図（EMG）など、複数の異なる信号源から収集されます。それらを一元的に解釈し、相関関係を分析するのは困難です。
	•	AIによる解決策: マルチモーダルデータ解析に特化したAIモデルを使用して、異なる種類の生体信号データを統合的に処理できます。たとえば、脳波と心拍の関係を同時に解析することで、より正確な生体状態の推測や、ストレスや集中度の高精度な評価が可能になります。ディープラーニングを用いることで、複数の信号の相互作用を学習し、統合的に解釈できるシステムを構築できます。

5. 個人差の影響. 

	•	課題: 生体信号は個人ごとの違いが大きいため、汎用的なモデルで正確に解析することが難しい場合があります。特に、異なる年齢や性別、体格などに応じて、信号パターンが大きく異なることがあります。
	•	AIによる解決策: 個人差に対応するために、パーソナライズドAIモデルを使用して、各ユーザーに合わせたフィッティングを行うことが可能です。転移学習や少量のデータを用いたファインチューニングを行うことで、個々のユーザーに最適化されたモデルを構築できます。これにより、ユーザーごとに高精度の解析が可能になり、個別の健康管理やモニタリングに役立ちます。
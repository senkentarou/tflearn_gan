# tflearnによるGANの構成

## 概要

 * tflearnでGAN(Generative Adversarial Nets)を実装
 * データセットはMNIST

## 内容物

 * scripts/train_gan_mnist.py: 学習用
 * scripts/evaluate_gan_mnist.py: 評価用(画像生成)
 * results/mnist_trained_50000_epoch_50_hidden_256.png
   (各中間層数256, 学習サンプル50000, エポック数50のときの結果)
 
## 環境

 * ubuntu 14.04 LTS
 * python 3.4.5 (virtualenv)
   * tensorflow >= 1.0.0 [Installing Tensorflow on Ubuntu](https://www.tensorflow.org/install/install_linux)
   * tflearn >= 0.3.0 [tflearn - github](https://github.com/tflearn/tflearn)

## 実行手順

 1. train_gan_mnist.pyを実行
  * 初回実行時は, mnistが../datasets/mnistにダウンロードされる.
 2. evaluate_gan_mnist.pyを実行
  * resultsディレクトリにresult.pngが生成されていることを確認する.

## その他
 * ネットワークの構造は, 元論文の[Generative Adversarial Nets](https://www.google.co.jp/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjXoMrnhq7TAhUTNrwKHdkvDM8QFggoMAA&url=https%3A%2F%2Farxiv.org%2Fabs%2F1406.2661&usg=AFQjCNEYi3th_8xRH0BIjxxmc-AM4lKdzA&sig2=-Vx5A51yVasuZweSj1L0mg)に従っている.
 * 区分線形ユニットや, 中間層の数, エポック数などはハイパーパラメータとして決定したため,  
   適切でない場合がある.

# Baysian-Linear-Regression
ベイズ線形回帰のフルスクラッチ実装

## How to use
\>\>\> docker-compose up  
(別のタブを開いて)  
\>\>\> docker exec -it baysian-lr /bin/bash  
\>\>\> python3 main.py x1 x2 x3 x4  

ここで，
x1: 観測データの発生分布の平均  
x2: 観測データの発生分布の分散  
x3: 観測データの発生分布からのサンプリング数  
x4: 近似式(多項式)の項の数  

## How to calculate
### はじめに
入力値を，
'''math
¥begin{eqnarray}
    {¥bf x}_n &=&
        ¥left(
            ¥begin{array}{c}
                x_{1}^{n} ¥¥
                x_{2}^{n} ¥¥
                ¥vdots ¥¥
                x_{m}^{n}
            ¥end{array}
        ¥right)．
¥end{eqnarray}
'''
重みを，
'''math
¥begin{eqnarray}
    {¥bf w} &=&
        ¥left(
            ¥begin{array}{c}
                w_1 ¥¥
                w_2 ¥¥
                ¥vdots ¥¥
                w_m
            ¥end{array}
        ¥right)，
¥end{eqnarray}
'''
とすると出力値は，
'''math
¥begin{eqnarray}
    y_n &=& {¥bf w}^{¥mathrm{T}} {¥bf x}_n + ¥epsilon _{n}，
¥end{eqnarray}
'''
と表現される．
ここで，$¥epsilon _{n}$はn番目の入力値に対する出力値の残差である．
今回残差は，
'''math
¥begin{eqnarray}
    ¥epsilon _{n} &¥sim& ¥mathcal{N}(¥epsilon _{n}|0,¥lambda^{-1})，
¥end{eqnarray}
'''
に従い発生すると仮定する．$¥lambda^{-1}$は精度であり，分散の逆数である．
これらを基に出力値$y_n$の確率分布を定式化すると，
'''math
¥begin{eqnarray}
    p(y_n|{¥bf x}_n , {¥bf w}) &=& ¥mathcal{N}(y_n|{¥bf w}^{¥mathrm{T}} {¥bf x}_n , ¥lambda^{-1})，
¥end{eqnarray}
'''
と表現することができる．

### 事後分布の算出
式(¥ref{eq: pd_of_y})の期待値を更新するには，観測データ(学習データ)から回帰モデルの重みを更新する必要がある．
したがって，
'''math
¥begin{eqnarray}
    ¥displaystyle 
    p({¥bf w}|{¥bf X} , {¥bf Y}) &=& ¥frac{p({¥bf Y}, {¥bf X}, {¥bf w})} {p({¥bf Y}|{¥bf X}) p({¥bf X})}，  ¥¥
                                 &=& ¥frac{p({¥bf Y}|{¥bf X}, {¥bf w}) p({¥bf X}) p({¥bf w})} {p({¥bf Y}|{¥bf X}) p({¥bf X})}，¥label{eq: rs} ¥¥
                                 &=& ¥frac{¥prod_{n=0}^N p({y_n}|{¥bf x}_n, {¥bf w}) p({¥bf w})} {p({¥bf Y}|{¥bf X})}， ¥label{eq: ex_pd}
¥end{eqnarray}
'''
を計算し，観測データセット${¥bf X}, {¥bf Y}$が与えられた時における重み$p({¥bf w})$の事後分布を算出する必要がある．
この時，式(¥ref{eq: rs})の分子は$p({¥bf X})$と$p({¥bf w})$が独立であり，分母は$p({¥bf Y})$と$p({¥bf X})$が従属の関係である．

ここで重要なのが，重み$p({¥bf w})$の事後分布を算出するには，重み$p({¥bf w})$の事後分布の形状を明らかにする必要がある．
そこで，式(¥ref{eq: ex_pd})に対し対数をとると，
'''math
¥begin{eqnarray}
    ¥displaystyle 
    ¥ln{p({¥bf w}|{¥bf X} , {¥bf Y})} &=& ¥ln{¥frac{¥prod_{n=0}^N p({y_n}|{¥bf x}_n, {¥bf w}) p({¥bf w})} {p({¥bf Y}|{¥bf X})}}，  ¥¥
                                      &=& ¥sum_{n=0}^N ¥ln{ p({y_n}|{¥bf x}_n, {¥bf w})} + ¥ln{p({¥bf w})} - ¥ln{{p({¥bf Y}|{¥bf X})}}， ¥¥
                                      &=& ¥sum_{n=0}^N ¥ln{ p({y_n}|{¥bf x}_n, {¥bf w})} + ¥ln{p({¥bf w})} + ¥mathrm{const.} ¥label{eq: const}，
¥end{eqnarray}
'''
が得られる．
式(¥ref{eq: const})の『const.』はconstantの略であり，任意定数であることを示している．
$¥ln{p({¥bf w}|{¥bf X} , {¥bf Y})}$は$¥ln{{p({¥bf Y}|{¥bf X})}}$の影響を受けないため，任意定数(const.)としている．
ここで$p({y_n}|{¥bf x}_n, {¥bf w})$は式(¥ref{eq: pd_of_y})を参照し，重み$p({¥bf w})$がM次元正規分布$¥mathcal{N}({¥bf w}|{¥bf m} , {¥bf ¥Lambda}^{-1})$に従うとすると，
'''math
¥begin{eqnarray}
    ¥displaystyle 
    ¥ln{p({¥bf w}|{¥bf X} , {¥bf Y})} &=& ¥sum_{n=0}^N ¥ln{¥mathcal{N}(y_n|{¥bf w}^{¥mathrm{T}} {¥bf x}_n , ¥lambda^{-1})} + ¥ln{¥mathcal{N}({¥bf w}|{¥bf m} , {¥bf ¥Lambda}^{-1})} + ¥mathrm{const.}，  ¥¥
                                      &=& -¥frac{1}{2} ¥sum_{n=0}^N ¥lambda (y_n - {¥bf w}^{¥mathrm{T}} {¥bf x}_n)^2 -¥frac{1}{2}({¥bf w} - {¥bf m})^{¥mathrm{T}} {¥bf ¥Lambda} ({¥bf w} - {¥bf m}) + ¥mathrm{const.}，¥label{eq: expansion}                                                 
¥end{eqnarray}
'''
となる．
式(¥ref{eq: expansion})を解き進めると，
'''math
¥begin{eqnarray}
    ¥displaystyle 
    ¥ln{p({¥bf w}|{¥bf X}, {¥bf Y})} &=& - ¥frac{1}{2} ¥{ {¥bf w}^{¥mathrm{T}} ( ¥lambda ¥sum_{n=0}^{N} {¥bf x}_n {¥bf x}_n^{¥mathrm{T}} + {¥bf ¥Lambda} ) {¥bf w}
                                         - 2 {¥bf w}^{¥mathrm{T}} ( ¥lambda ¥sum_{n=0}^{N} y_n {¥bf x}_n + {¥bf ¥Lambda} {¥bf m} ) ¥} + ¥mathrm{const.}，¥label{eq: solution}  ¥¥                                            
¥end{eqnarray}
'''
が得られる(詳細は後日upします)．
これはM次元正規分布の対数と同じ，つまり{¥bf w}の事後分布は{¥bf w}の事前分布と同じくM次元正規分布として書けることがわかった．
したがって，
'''math
¥begin{eqnarray}
    ¥displaystyle 
    p({¥bf w}|{¥bf X} , {¥bf Y}) &=& ¥mathcal{N}({¥bf w}|{¥bf ¥hat{m}}, {¥bf ¥hat{¥Lambda}})，¥¥
    ¥ln{p({¥bf w}|{¥bf X} , {¥bf Y})} &=&  - ¥frac{1}{2} ¥{ {¥bf w}^{¥mathrm{T}} {¥bf ¥hat{¥Lambda}} {¥bf w} - 2 {¥bf w}^{¥mathrm{T}} {¥bf ¥hat{¥Lambda}} {¥bf ¥hat{m}} ¥} + ¥mathrm{const.}， ¥label{eq: ex_pd_of_st}                                                
¥end{eqnarray}
'''
から，式(¥ref{eq: solution})と式(¥ref{eq: ex_pd_of_st})より，
'''math
¥begin{eqnarray}
    ¥displaystyle 
    {¥bf ¥hat{¥Lambda}} &=& ¥lambda ¥sum_{n=0}^{N} {¥bf x}_n {¥bf x}_n^{¥mathrm{T}} + {¥bf ¥Lambda}，¥¥
    {¥bf ¥hat{m}} &=& {¥bf ¥hat{¥Lambda}}^{-1} ( ¥lambda ¥sum_{n=0}^{N} y_n {¥bf x}_n + {¥bf ¥Lambda} {¥bf m} )，
¥end{eqnarray}
'''
が得られる．
得られた${¥bf ¥hat{m}}$，${¥bf ¥hat{¥Lambda}}$が$p({¥bf w})$の事後分布の期待値と精度(分散)である．
観測データセットにより，これらを更新し，更新後のパラメータを用いた予測分布を算出することで，
新規入力値に対する出力値の期待値と精度(分散)を獲得する．

### 予測分布の算出
coming soon

## Results
![result](https://github.com/rrrrind/Baysian-Linear-Regression/blob/main/workspace/results/result.gif)

## References
- [3.5：線形回帰の例【緑ベイズ入門のノート】](https://www.anarchive-beta.com/entry/2020/11/12/100521)
- [注意すべき行列の性質](https://oguemon.com/study/linear-algebra/matrix-notice/)
- [内積の２乗](https://ameblo.jp/accade/entry-11899474992.html)
- [二次形式](https://mathwords.net/vecmatseki)
- [転置行列の性質（＋α）](http://web.wakayama-u.ac.jp/~wuhy/am3.pdf)
- [Woodburyの恒等式](https://mathtrain.jp/woodbury)
- [Sherman-Morrisonの公式](http://ibisforest.org/index.php?Sherman-Morrison%E3%81%AE%E5%85%AC%E5%BC%8F)
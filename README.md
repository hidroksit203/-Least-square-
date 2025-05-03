# -Least-square-

# Doğrusal Regresyon (Linear Regression) ile Modelleme ve Karşılaştırmalı Yöntem İncelemesi  
> En küçük kareler ve gradyan inişi yöntemleriyle doğrusal regresyonun saf Python ve scikit-learn kullanılarak incelenmesi

---

## 1. Temel Kavramlar

### 1.1 Doğrusal Regresyon  
Doğrusal regresyon, bir veya daha fazla bağımsız değişken ile bağımlı değişken arasındaki doğrusal ilişkiyi modelleyen istatistiksel bir tekniktir.

**Basit haliyle formülasyon:**  
```
y = β₀ + β₁x + ε
```
- `y`: hedef değişken  
- `x`: girdi/özellik  
- `β₀`: y-kesişim (bias)  
- `β₁`: x'in katsayısı (eğim)  
- `ε`: hata terimi

### 1.2 Matris Formunda Model  
Çoklu doğrusal regresyon matris şeklinde şu şekilde yazılır:
```
y = Xβ + ε
```
En küçük kareler tahminiyle β değerleri:
```
β = (XᵗX)⁻¹Xᵗy
```

### 1.3 Maliyet Fonksiyonu (Cost Function)  
Modelin ne kadar iyi olduğunu değerlendirmek için ortalama karesel hata (MSE) kullanılır:

```
J(β) = (1/n) * Σ (yᵢ - (β₀ + β₁xᵢ))²
```

---

## 2. Regresyon Yöntemleri

### 2.1 Least Squares (Kapalı Form Çözüm)
- `numpy.linalg.inv` ve `dot` ile `(XᵗX)^-1 * Xᵗy` hesaplanır.
- Doğrudan çözüm sunar ve küçük boyutlu veri setlerinde oldukça etkilidir.

### 2.2 Gradient Descent (İteratif Optimizasyon)
- Parametreler her iterasyonda güncellenir:
```
θ := θ - α * ∇J(θ)
```
- Öğrenme oranı (α) ve epoch sayısı çok kritiktir.

### 2.3 Scikit-learn LinearRegression
- `LinearRegression()` sınıfı kullanılarak:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
```
- Model `.coef_` ve `.intercept_` değerleri ile doğrudan erişilir.

---

## 3. Model Karşılaştırması

| Yöntem             | Hesaplama | Hız | Yorum |
|--------------------|-----------|-----|-------|
| Kapalı Form        | Doğrudan  | ⚡⚡⚡ | Matematiksel olarak net |
| Gradyan İnişi      | İteratif  | ⚡⚡  | Ayar gerektirir |
| Scikit-learn       | Otomatik  | ⚡⚡⚡ | Kullanımı çok kolay |

---

## 4. Uygulama ve Kodlama Detayları

### 4.1 Veri Seti  
Veri yaş ve maaş gibi iki boyutlu örneklerle oluşturulmuştur:
```csv
Age,Salary
22,2500
25,2700
30,3200
...
```

### 4.2 Kod Akışı
- Veri Pandas ile okunur.
- Özellik vektörü ve hedef vektör oluşturulur.
- Eğitim/test ayrımı yapılır.
- Üç yöntem uygulanır.
- MSE ve tahminler görselleştirilir.

---

## 5. Görselleştirme ve Sonuçlar

### 5.1 Regresyon Doğruları
Matplotlib kullanılarak:
- Eğitim verisi
- Modelin tahmin eğrileri

### 5.2 Gradient Descent Cost Eğrisi
- Epoch’a göre MSE değişimi çizilir.

---

## 6. Teorik ve Pratik Değerlendirme

- Kapalı form çözüm genellikle doğruluğu garantiler.
- Gradient descent öğrenme oranına ve veri ölçeğine duyarlıdır.
- Scikit-learn uygulaması zaman kazandırır ancak modeli “anlama” sürecini atlar.

---

## 7. Yöntemler ve Uygulama Alanları

| Yöntem                 | Temel İşlem                         | Nerede Kullanılır                 |
|------------------------|--------------------------------------|-----------------------------------|
| En Küçük Kareler       | Kapalı form çözüm                    | Küçük veri setleri, yorumlama     |
| Gradyan İnişi          | İteratif optimizasyon                | Büyük veri, çevrimiçi öğrenme     |
| scikit-learn Regression| Otomatik model tahmini               | Hızlı prototipleme, üretim sistemleri |

---

## 8. Kaynakça

- https://github.com/patrickloeber/MLfromscratch/
- https://github.com/chasinginfinity/ml-from-scratch
- https://www.geeksforgeeks.org/solving-linear-regression-without-using-sklearn-and-tensorflow/
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

---

## 9. Least Squares Estimation Yönteminin Derinlemesine İncelemesi

### 9.1 Giriş
Least Squares Estimation (LSE), regresyon analizinde en yaygın kullanılan yöntemlerden biridir. Temel amacı, gözlemlenen değerler ile modelin tahmin ettiği değerler arasındaki farkların karelerinin toplamını en aza indiren regresyon katsayılarını (β) bulmaktır.

### 9.2 Matematiksel Temel
Verilen bir veri setinde n adet gözlem ve p adet açıklayıcı değişken varsa, modeli matris formunda şu şekilde ifade ederiz:

```
y = Xβ + ε
```

- `y`: n × 1 boyutunda hedef (bağımlı değişken) vektörü  
- `X`: n × p boyutunda açıklayıcı değişkenler (özellikler) matrisi  
- `β`: p × 1 boyutunda model parametreleri vektörü  
- `ε`: hata (residual) vektörü

Amaç, aşağıdaki maliyet fonksiyonunu minimize etmektir:

```
J(β) = (y - Xβ)ᵀ(y - Xβ)
```

Bu ifadeyi türevleyip sıfıra eşitleyerek β'nın kapalı form çözümünü elde ederiz:

```
∇J(β) = -2Xᵀ(y - Xβ) = 0
```

Buradan:

```
XᵀXβ = Xᵀy  →  β = (XᵀX)⁻¹Xᵀy
```

Bu formül **normal denklemler** olarak da bilinir.

### 9.3 Varsayımlar
LSE yönteminin geçerli sonuçlar üretebilmesi için bazı istatistiksel varsayımların sağlanması gerekir:

1. **Lineerlik**: Modelin bağımlı değişken ile açıklayıcı değişkenler arasında doğrusal bir ilişki kurduğu varsayılır.
2. **Bağımsızlık**: Gözlemler birbirinden bağımsız olmalıdır.
3. **Homoskedastisite**: Hataların varyansı sabittir.
4. **Normallik**: Hatalar normal dağılıma sahiptir.
5. **Çoklu doğrusallık olmaması**: Özellikler arasında yüksek korelasyon bulunmamalıdır.

### 9.4 Avantajları
- Kapalı form çözümü ile doğrudan sonuç verir.
- Küçük ve orta ölçekli veri setlerinde yüksek verimlilik sağlar.
- Yorumlanabilir parametreler üretir.

### 9.5 Dezavantajları
- `XᵀX` matrisinin tersinin alınması hesaplama açısından maliyetlidir.
- Büyük veri setlerinde veya çok boyutlu yapılarda hesaplama yükü artar.
- Özellikler arasında korelasyon varsa sonuçlar kararsız olabilir (multicollinearity).

### 9.6 Python'da Uygulama
```python
import numpy as np

# Özellik ve hedef vektörleri
X = np.array([[1, x] for x in ages])  # Bias eklenmiş hali
y = np.array(salaries)

# β hesapla
beta = np.linalg.inv(X.T @ X) @ X.T @ y
```

---


## 10. Least Squares Modeli Matematiksel Olarak Nasıl Çalışır?

Least Squares (En Küçük Kareler) yöntemi, doğrusal regresyon modelinde bilinmeyen parametreleri (β katsayıları) tahmin etmek için kullanılır. Temel hedef, modelin tahminleri ile gözlemlenen değerler arasındaki farkların karelerinin toplamını minimize etmektir.

### 10.1 Modelin Matematiksel Tanımı

Veri noktaları aşağıdaki gibi tanımlanır:

- `X` ∈ ℝⁿˣᵖ: Özellik matrisi (n gözlem, p özellik)
- `y` ∈ ℝⁿ: Gözlenen sonuçlar
- `β` ∈ ℝᵖ: Öğrenilecek parametreler
- `ε` ∈ ℝⁿ: Hatalar (residuals)

Model:

```
y = Xβ + ε
```

### 10.2 Amaç Fonksiyonu (Cost Function)

Least Squares, şu fonksiyonu minimize etmeye çalışır:

```
J(β) = ||y - Xβ||² = (y - Xβ)ᵗ(y - Xβ)
```

Bu, model tahmini ile gerçek gözlemler arasındaki kare farkların toplamıdır.

### 10.3 Türev ve Optimum Çözüm

Fonksiyonun minimum noktasını bulmak için türev alınır:

```
∇J(β) = -2Xᵗ(y - Xβ) = 0
```

Bu denklemden:

```
XᵗXβ = Xᵗy
```

Ve nihayet kapalı form çözüm:

```
β = (XᵗX)⁻¹Xᵗy
```

Bu formül, verilen veri seti için "en iyi uyum sağlayan" parametre setini verir.

### 10.4 Yorumsal Açıklama

- `XᵗX`: Özellikler arasındaki korelasyon yapısını temsil eder.
- `(XᵗX)⁻¹Xᵗ`: Moore–Penrose pseudo-inverse işlemcisidir.
- Model deterministik bir şekilde eğitilir, dolayısıyla aynı veri setiyle her zaman aynı β sonuçları üretir.

### 10.5 Varsayımlar

Modelin doğru çalışabilmesi için bazı istatistiksel varsayımlar vardır:

1. Doğrusallık
2. Sabit varyans (homoskedastisite)
3. Normal hata dağılımı
4. Özellikler arası çoklu doğrusal bağlantının olmaması

Bu varsayımlar sağlanmadığında modelin tahmin gücü azalabilir.

---


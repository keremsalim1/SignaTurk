# Sunum Demo Kelimeleri (Altın Set)

Avatar el-derinliği düz olduğu için (MediaPipe el-z güvenilmez) **ince parmak
konfigürasyonuna** bağlı işaretler avatarda birbirine benzer görünür. Buna karşılık
**kol/bilek genliği yüksek** işaretler net ve inandırıcı render edilir. Aşağıdaki
sıralama 226 kelimenin kol+dirsek+bilek hareket genliğine göredir.

## ✅ Önerilen demo kelimeleri (yüksek hareket, kalite ≥ 96)

| Kelime | Genlik | Kalite | Not |
|---|---|---|---|
| hastane | 1.66 | 98.0 | iki el, geniş hareket |
| seytan | 1.57 | 98.3 | iki el |
| kavsak | 1.56 | 98.5 | iki el |
| orman | 1.55 | 96.8 | iki el |
| yildiz | 1.53 | 98.4 | iki el |
| kolonya | 1.42 | 98.7 | iki el |
| bilgi_vermek | 1.37 | 98.5 | iki el |
| fil | 1.32 | 98.1 | iki el, akılda kalıcı |
| pencere | 1.26 | 98.8 | iki el |
| gol | 1.21 | 98.9 | iki el |
| havlu | 1.18 | 98.5 | iki el |
| yatak | 1.18 | 98.4 | iki el |
| memnun_olmak | 1.16 | 98.8 | iki el |
| doktor | 1.06 | 99.0 | iki el |
| yardim | 1.02 | 98.7 | iki el |

> `tavan` (genlik 1.83) en hareketli kelime; güvenle kullanılabilir (eskiden onunla aynı
> veriyi paylaşan artık `class_191` anahtarı silindi).

## 🟢 GÜVENLİ "kolay" set (kafadan uzak + el-teması yok + net) — GÖRSEL DOĞRULANDI

Sunumda en az riskli işaretler: eller kafaya yaklaşmıyor, gövdeye/birbirine değmiyor,
hareket net, kalite ~99. Avatar üzerinde render edilip gözle teyit edildi.

| Kelime | Görünüm | Not |
|---|---|---|
| onlar | ✅ en iyi | sağ el yukarı net işaret-eli |
| pantolon | ✅ çok güvenli | eller belde, doğal yumruk |
| cocuk | ✅ temiz | iki el ayrı |
| kemer | ✅ temiz | iki el ayrı |
| yakin | ✅ iyi | sağ el göğüs önü (hafif gövdeye yakın) |
| getirmek | ✅ temiz | baştan uzak, net kol/bilek hareketi |
| ataturk | ✅ temiz | göğüs hizasında, eller ayrık |
| hali | ✅ temiz | baştan uzak, iki el ayrı |
| evet | ◑ sınırda | açık el yüz hizasına çıkıyor (kafaya en yakın) |

### ➕ Sete eklenen güvenli kelimeler (metrik + doğru-işaret doğrulandı)

Aşağıdakiler aynı güvenlik zarfına (eller birbirine değmiyor, hiçbir karede el
burun hizasının ÜSTÜNE çıkmıyor, hareket net) uyuyor; `source_video` AUTSL
SignList'te doğru sınıfa eşleşiyor (eşleme-kayması mağduru DEĞİL) ve el tespiti
30/30 tam. _Not: bunlar geometrik metrik ve doğru-işaret çapraz-kontrolüyle
doğrulandı; yukarıdaki 9'un aksine henüz avatarda gözle render edilmedi — son bir
göz teyidi önerilir._

| Kelime | El açıklığı (min) | Baş mesafesi (min) | Hareket genliği | Not |
|---|---|---|---|---|
| kacmak | 0.174 | 0.181 | 0.464 | en net hareket (tüm sette en yüksek genliklerden); baştan en uzak |
| tuvalet | 0.161 | 0.135 | 0.462 | iki el, çok net hareket; baş mesafesi `evet`/`hali` ile aynı seviyede |
| anne | 0.152 | 0.149 | 0.353 | iki el ayrı, `getirmek` ile benzer profil; el yüze değmiyor |

> Diğer güvenli adaylar (kalite≥96, kafadan/gövdeden uzak): `serbest, yarin,
> gecmis, hakli, dusman`. Not: işaret dili vücut alanını kullandığı için hiçbiri tam
> izole değil; bunlar göreceli en temizleri.

## ⚠️ Sunumda kaçınılması gerekenler (statik / parmak-bağımlı, düşük genlik)

`masa`, `pazartesi`, `sut`, `acikmak`, `nasil`, `para`, `kotu`, `iyi`,
`ogretmen`, `ayni`, `ben`, `makas` — bilek neredeyse sabit, ayırt edicilik
ince parmak şeklinde → avatarda zayıf görünür.

## 🔧 Veri temizliği (TAMAMLANDI)

`model/landmarks.json` içindeki artık **`class_191`** anahtarı **silindi** (gerçek kelime
değildi, `tavan` ile aynı videoya işaret ediyordu, kelime grid'inde "class_191" çipi olarak
görünüyordu). Kelime sayısı 227 → **226** oldu (AUTSL sınıf sayısıyla birebir).
Yedek: `model/landmarks.json.bak_class191`.

## ℹ️ Neden bazı işaretler benzer / el-şekli yok?

El-şekli **veride yok** (AUTSL videoları el için düşük çözünürlüklü; iki farklı çıkarıcı —
MediaPipe ve RTMW — da düz/açık el veriyor). Parmaklar oynar ama belirgin handshape oluşmaz.
Bu yüzden yalnızca el-şekliyle ayrışan işaretler (yukarıdaki "kaçınılması gerekenler") avatarda
benzer görünür. Ayrıntı ve "daha iyisi için ne yapılmalı" için bkz. `PROJECT_STATUS.md` →
"Animasyon Retarget Revizyonu + Tavan Analizi".

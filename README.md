# futur — BTC Futures Signal System

Signal trading BTCUSDT futures berbasis riset 10 paper (lihat `~/Downloads/Paper_BTC_Futures_Strategy`).

## Strategi Default

Pipeline default aktif: **VMA(5,120) + ATR filter + RF filter ON (N=3, threshold 0.5)**; onchain OFF; GARCH OFF.

- **Signal**: VMA(5,120) cross di timeframe 4H (paper 05 — kombinasi terbaik dari heatmap)
- **Vol filter**: skip signal jika ATR% > 6% (paper 01 — extreme vol = liquidation cascade)
- **RF confidence filter (DEFAULT ON, N=3, th=0.5)**: Random Forest classifier walk-forward expanding window (paper 07 — Jevtic et al. 2022) memprediksi arah return N=3 bar (~12 jam) ke depan dari fitur masa lalu & on-chain tanpa look-ahead. Signal VMA hanya dirilis jika RF mengonfirmasi arah (LONG: prob >= 0.5; SHORT: prob <= 0.5)
- **On-chain filter (opsional / OFF by default)**: filter tren on-chain via rolling z-score hash rate dan n-unique-addresses (paper 02 — Nagula & Alexakis 2022); suppress LONG jika keduanya di bawah threshold negatif dan suppress SHORT jika keduanya di atas threshold positif
- **GARCH vol filter (opsional / OFF by default)**: model GARCH(1,1) pada return 4H, skip signal jika conditional volatility forecast 1-step melebihi persentil historis (paper 09 — Ciaian et al. 2018)
- **Risk layer**: leverage hard-cap 3X, risk 2% equity/trade, SL = 2×ATR, TP = 4×ATR (RR 1:2)
- **Exit**: SL/TP intrabar (SL dicek duluan = konservatif) atau signal berlawanan

## Backtest (2019-01 → 2026-10, 17.107 bar 4H, fee taker 0.05%)

| Varian | Trades | Win Rate | Return | Max DD | Sharpe |
|--------|--------|----------|--------|--------|--------|
| **Baseline (ATR only)** | 343 | 35.3% | +238.2% | -25.3% | 1.00 |
| **+Onchain filter** | 291 | 38.5% | +251.1% | -21.6% | 1.07 |
| **+GARCH filter** | 331 | 35.0% | +203.6% | -26.9% | 0.93 |
| **+Onchain +GARCH** | 282 | 38.3% | +222.5% | -21.4% | 1.02 |
| **+RF filter (DEFAULT: N=3, th=0.5)** | **95** | **51.6%** | **+196.8%** | **-10.0%** | **1.36** |
| **+Onchain +RF (N=3, th=0.5)** | 82 | 53.7% | +164.9% | -8.1% | 1.30 |
| *+RF filter (N=6, th=0.5 - lama)* | 100 | 44.0% | +97.1% | -9.8% | 0.89 |
| *+Onchain +RF (N=6, th=0.5 - lama)* | 88 | 44.3% | +74.8% | -7.9% | 0.78 |

Konfigurasi terbaik terpilih sebagai default baru adalah **+RF filter (N=3, th=0.5)**:
- Sharpe ratio tertinggi (1.36)
- Win rate naik drastis dari baseline 35.3% menjadi 51.6%
- Drawdown terpangkas lebih dari separuh (-25.3% -> -10.0%)
- Eliminasi whipsaw jangka pendek tanpa bias bullish secular berlebih yang sebelumnya memotong trade SHORT pada N=6

## RF Walk-Forward Cache (`data/rf_cache.json`)

Pelatihan ulang Random Forest walk-forward expanding window pada ~17.000 bar memerlukan waktu komputasi intensif. Untuk performa scanner dan backtest:
- Probabilitas `rf_prob_long` disimpan ke `data/rf_cache.json` dengan key `open_time -> prob`.
- Validasi integritas berbasis `param_hash` (`n_bars`, `min_train`, `retrain_every`, `n_estimators`, `max_depth`, `min_samples_leaf`, `random_state`). Jika parameter diubah, cache otomatis invalid dan dilatih ulang.
- Pelatihan inkremental: chunk yang sudah ada dalam cache dilewati (0s), hanya bar/chunk baru yang dilatih dan di-append. Scanner run kedua dan seterusnya selesai dalam <4 detik.

## Files

| File | Fungsi |
|------|--------|
| `data.py` | Fetch + cache klines BTCUSDT 4H (data-api.binance.vision, spot proxy perp — fapi di-block ISP). Cache incremental di `data/`, TTL 15 menit. |
| `onchain.py` | Fetch + cache on-chain BTC (hash rate & unique addresses dari blockchain.info charts API gratis), hitung rolling z-score dan resample ke 4H. |
| `garch_vol.py` | Fit GARCH(1,1) pada log-return 4H via `arch` library, forecast 1-step conditional volatility. |
| `rf_filter.py` | Fit RandomForest walk-forward expanding window (paper 07, default N=3), prediksi arah N bar ke depan dari fitur masa lalu & on-chain tanpa look-ahead. Mendukung caching ke `data/rf_cache.json`. |
| `strategy.py` | VMA cross + ATR + On-chain filter + GARCH filter + RF confidence filter (default ON: N=3, th=0.5) → kolom `sig` (+1/-1/0) dan `vma_sig`. |
| `backtest.py` | Backtest engine + metrics. Default: +RF (N=3, th=0.5). Mendukung flag `--no-rf`, `--onchain`, `--garch`, `--both`. Hasil trades → `backtest_trades.json` |
| `compare.py` | Script pembanding performa 6 varian backtest (default N=3). |
| `notifier.py` | Mengirim notifikasi Telegram via Bot API HTTPS (`send_telegram`), membaca kredensial dari `notifier_config.json`. |
| `scanner.py` | Live scanner: cek bar 4H terakhir yang closed, konfirmasi arah via RF N=3. Mendukung flag `--notify` untuk alert Telegram. |

## Usage

```bash
.venv/Scripts/python backtest.py                         # backtest default (+RF N=3, th=0.5)
.venv/Scripts/python backtest.py --no-rf                 # backtest baseline (ATR only)
.venv/Scripts/python backtest.py --onchain               # backtest default + onchain filter
.venv/Scripts/python compare.py                          # perbandingan 6 varian (default N=3)
.venv/Scripts/python compare.py --rf-n 6                 # perbandingan dengan horizon N=6 lama
.venv/Scripts/python scanner.py                          # live scanner candle 4H closed terakhir (print saja)
.venv/Scripts/python scanner.py --notify                 # live scanner dengan alert Telegram
.venv/Scripts/python notifier.py                         # uji coba koneksi Telegram bot
```

## Telegram Notifier

Sistem mendukung pengiriman alert signal ke Telegram melalui Bot API HTTPS (`https://api.telegram.org/bot<TOKEN>/sendMessage`).

### Konfigurasi Kredensial
File `notifier_config.json` disimpan di direktori root project:
```json
{
  "telegram": {
    "bot_token": "<BOT_TOKEN>",
    "chat_id": "<CHAT_ID>"
  }
}
```

> **Catatan Keamanan:**
> File `notifier_config.json` memuat kredensial rahasia bot Telegram. File ini dimasukkan ke `.gitignore` agar tidak ter-commit ke version control. Jangan pernah membagikan atau mem-print isi token ke log publik.

### Rekomendasi Cron / Otomasi (100% Gratis Tanpa Laptop Nyala)

Project ini dilengkapi dengan **GitHub Actions** (`.github/workflows/scanner.yml`) yang berjalan otomatis setiap 4 jam di cloud GitHub/Microsoft:
- Jadwal: Menit ke-5 setelah candle 4H close (`00:05`, `04:05`, `08:05`, `12:05`, `16:05`, `20:05` UTC / `07:05`, `11:05`, `15:05`, `19:05`, `23:05`, `03:05` WIB).
- Menggunakan `actions/cache` untuk menyimpan cache data klines & RF probability sehingga eksekusi sangat cepat (<20 detik).

**Cara Mengaktifkan di GitHub:**
1. Buka repository kamu di GitHub (`kullal/futures`).
2. Masuk ke **Settings** > **Secrets and variables** > **Actions**.
3. Buat dua **Repository Secrets**:
   - `TELEGRAM_BOT_TOKEN`: Token bot Telegram kamu (contoh: `8663980044:...`)
   - `TELEGRAM_CHAT_ID`: ID chat Telegram kamu (contoh: `1075969063`)
4. Masuk ke tab **Actions** di GitHub, pilih **BTC Futures 4H Scanner**, lalu klik **Run workflow** untuk uji coba manual pertama kali.

Jika ingin menjalankan lokal di Windows tanpa GitHub, gunakan Windows Task Scheduler (`cron_scan.py`).

## Roadmap

- [x] Backtest engine + baseline VMA(5,120)
- [x] Integrasi on-chain features filter (hash rate + active addresses, paper 02)
- [x] Integrasi GARCH(1,1) conditional volatility filter (paper 09)
- [x] ML filter (RF confidence, paper 07) dengan horizon N=3 sebagai default
- [x] Cache disk RF walk-forward untuk percepatan scanner & backtest
- [x] Integrasi RF filter ke live scanner
- [x] Telegram notifier untuk alert signal
- [x] Cron tiap candle close 4H (00/04/08/12/16/20 UTC +1 menit)
- [ ] Outcome tracker (signal → TP/SL hit, bahan evaluasi)

## Catatan

- Data spot dipakai sebagai proxy perpetual (fapi.binance.com diblokir ISP). Basis spot-perp kecil tapi funding rate TIDAK termasuk dalam backtest.
- Win rate rendah (~35% baseline) itu normal untuk trend-following RR 1:2 — dengan RF filter N=3 win rate naik menjadi 51.6%.
- Backtest ≠ masa depan: belum slippage, belum funding, satu pair saja.

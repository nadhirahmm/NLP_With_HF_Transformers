<h1 align="center"> 🧠 NLP With HuggingFace Transformers 🧠 </h1>
<p align="center"> Explore the power of state-of-the-art Natural Language Processing (NLP) using the Hugging Face Transformers library, all in one place! This repository covers various essential NLP tasks including Sentiment Analysis, Named Entity Recognition (NER), Question Answering, Summarization, and Translation.</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">

</div>

<h2 align="center"> Analisis HuggingFace Transformers NLP Pipelines </h2>

---


**1. Classification Task**

```
from transformers import pipeline
classifier = pipeline("zero-shot-classification")
```
Kode ini mengimpor fungsi `pipeline` dari pustaka `transformers` dan menginisialisasi pipeline untuk tugas zero-shot classification. Dengan zero-shot classification, model dapat mengklasifikasikan teks ke dalam label-label tertentu tanpa memerlukan pelatihan khusus pada label tersebut. Dalam hal ini, pipeline menggunakan model default `facebook/bart-large-mnli`, yang dirancang untuk tugas zero-shot classification dengan performa baik.


```
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business", "technology", "science"],
    multi_label=True
)
```
Fungsi `classifier` menerima input teks "This is a course about the Transformers library" dan label kandidat yang meliputi: `"education"`, `"politics"`, `"business"`, `"technology"`, dan `"science"`. Parameter `multi_label=True` memungkinkan model memberikan lebih dari satu label untuk teks tersebut dengan skor tertentu sehingga memungkinkan klasifikasi yang lebih fleksibel, terutama untuk kasus di mana teks dapat masuk ke beberapa kategori.

Output dari classifier memberikan hasil dalam bentuk skor untuk setiap label. Berikut ini adalah rincian skor untuk masing-masing label:

```
'technology': 0.28135383129119873
'education': 0.25892961025238037
'science': 0.001862368662841618
'business': 0.00049971270779798925
'politics': 9.32903581284314e-05
```
Skor-skor ini menunjukkan probabilitas atau keyakinan model terhadap setiap label. Skor tertinggi adalah untuk label `'technology'` dan `'education'`, yang masuk akal mengingat teksnya berbicara tentang kursus terkait pustaka Transformers, yang berhubungan erat dengan teknologi dan pendidikan. Label lain, seperti `'science'`, `'business'`, dan `'politics'`, memiliki skor yang sangat rendah, menunjukkan bahwa model tidak terlalu yakin bahwa teks tersebut relevan dengan label-label tersebut.

---
**2. Text Generation**

```
classifier = pipeline("zero-shot-classification")
classifier(
    "Stocks are experiencing a significant downturn.",
    candidate_labels=['education', 'economic bubble', 'politics'],
)
```
Fungsi `classifier` digunakan untuk mengklasifikasikan teks "Stocks are experiencing a significant downturn." (Saham mengalami penurunan signifikan). Label kandidat yang disediakan adalah: `'education'`, `'economic bubble'`, `'politics'`. Dengan kata lain, kita ingin mengetahui apakah model menganggap teks tersebut berkaitan dengan salah satu dari tiga label ini.

Berdasarkan skor yang dihasilkan, model memprediksi bahwa teks tersebut paling mungkin terkait dengan `'economic bubble'`, kemudian diikuti oleh `'politics'`, dan terakhir `'education'`. Hasil ini logis mengingat bahwa teks berkaitan dengan pasar saham dan memiliki implikasi ekonomi yang kuat.

---
**3. Fill-Mask Task**

```
from transformers import pipeline
unmasker = pipeline("fill-mask")
```
Kode ini mengimpor fungsi `pipeline` dari pustaka `transformers` dan menginisialisasi pipeline untuk tugas *fill-mask*. Dalam *fill-mask* task, model akan menggantikan token `<mask>` dengan kata-kata yang paling sesuai berdasarkan konteks kalimat. Pipeline ini menggunakan model `distilroberta-base` secara default.

```
unmasker("This course will teach you all about <mask> models.", top_k=5)
```
Teks yang diberikan mengandung token `<mask>`, yang menandakan bagian teks yang ingin diprediksi oleh model. Dalam kalimat ini, tujuan model adalah untuk mengganti <mask> dengan kata yang paling sesuai sehingga kalimat tersebut tetap memiliki makna logis. Parameter `top_k=5` menunjukkan bahwa model akan memberikan 5 prediksi teratas untuk token `<mask>` berdasarkan konteks.

Berdasarkan skor, kita dapat menyimpulkan bahwa model mengidentifikasi `"mathematical"` sebagai pengganti `<mask>` yang paling relevan dalam konteks kalimat ini. Kata-kata alternatif seperti `"computational"` dan `"predictive"` juga dipertimbangkan, tetapi dengan keyakinan lebih rendah. Hal ini menunjukkan bahwa model memahami kalimat tersebut sebagai sesuatu yang berhubungan dengan model matematis atau komputasional.

---
**4. NER - Exploring Non-Grouped**
```
from transformers import pipeline
ner = pipeline("ner") #, grouped_entities=True)
```
Kode ini mengimpor fungsi `pipeline` dari pustaka `transformers` dan menginisialisasi pipeline untuk tugas Named Entity Recognition (NER). Pipeline ini menggunakan model default `dbmdz/bert-large-cased-finetuned-conll03-english` yang telah dilatih untuk mengenali entitas seperti orang, lokasi, organisasi, dan lainnya dalam teks berbahasa Inggris.

Argumen `grouped_entities=True` di-comment, yang berarti model akan menampilkan setiap token entitas secara terpisah, bukan sebagai entitas yang dikelompokkan.

Output dari pipeline menunjukkan entitas yang terdeteksi beserta informasi seperti:

`entity`: Jenis entitas yang terdeteksi, dalam hal ini I-PER, yang berarti "Individual - Person" atau entitas orang.

`score`: Skor keyakinan model terhadap prediksi tersebut, antara 0 dan 1. Skor mendekati 1 menunjukkan keyakinan yang tinggi.

`word`: Token atau kata yang teridentifikasi sebagai entitas.

`start` dan `end`: Posisi indeks karakter awal dan akhir untuk entitas dalam teks asli.

---
**5. Question-Answering**
```
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
```
Kode ini mengimpor fungsi `pipeline` dari pustaka `transformers` dan menginisialisasi pipeline untuk tugas *question answering* (QA) menggunakan model `bert-large-uncased-whole-word-masking-finetuned-squad`. Model ini adalah varian besar dari BERT yang telah dilatih secara khusus pada dataset SQuAD (Stanford Question Answering Dataset) dan sangat cocok untuk tugas QA.
```
context = "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy."
question = "What is photosynthesis?"
```
Dalam kode ini, `context` menyediakan teks atau informasi yang relevan dengan topik, yang menjelaskan tentang fotosintesis. `question` menyimpan pertanyaan "What is photosynthesis?" yang akan dijawab oleh model berdasarkan informasi di dalam konteks.

---
**6. Sentiment Analysis**
```
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
```
Kode ini mengimpor fungsi `pipeline` dari pustaka `transformers` dan menginisialisasi pipeline untuk tugas *sentiment analysis*. Pipeline ini menggunakan model `distilbert-base-uncased-finetuned-sst-2-english` secara default, yang dilatih untuk mengklasifikasikan sentimen sebagai positif atau negatif pada dataset SST-2 (Stanford Sentiment Treebank).
```
classifier("It was just perfect if only it didn’t fail so much!")
```
Model diberikan teks yang mengandung sarkasme: "It was just perfect if only it didn’t fail so much!". Teks ini terlihat ambigu karena memulai dengan pujian ("just perfect") tetapi kemudian diikuti dengan kritik ("if only it didn’t fail so much!"). Secara logis, kalimat ini berkesan negatif karena mengandung sarkasme.
```
[{'label': 'POSITIVE', 'score': 0.9998670816421509}]
```
Meskipun kalimat mengandung sarkasme, model menganggapnya sebagai sentimen positif. Hal ini terjadi karena model tidak terlatih untuk mengenali sarkasme atau konteks ambigu dalam teks. Model hanya melihat kata-kata seperti "perfect" yang biasanya berasosiasi dengan sentimen positif sehingga gagal menangkap makna sarkastis yang dimaksudkan dalam kalimat.

---
**7. Summarization**
```
from transformers import pipeline
summarizer = pipeline("summarization")
```
Kode ini mengimpor fungsi `pipeline` dari pustaka `transformers` dan menginisialisasi pipeline untuk tugas *summarization* atau peringkasan teks. Pipeline ini menggunakan model default yang dirancang untuk menghasilkan ringkasan singkat dari teks yang lebih panjang.

Model summarization berfungsi untuk menyaring poin-poin utama dari teks panjang, sehingga informasi kunci dapat disajikan dalam bentuk yang lebih ringkas dan mudah dibaca. Ringkasan yang dihasilkan model biasanya menekankan masalah utama (seperti kurangnya minat generasi muda dalam pertanian) dan beberapa solusi atau persepsi yang perlu diubah.

---
**8. Translation**
```
from transformers import pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
```
Kode ini mengimpor fungsi `pipeline` dari pustaka `transformers` dan menginisialisasi pipeline untuk tugas translation (penerjemahan) menggunakan model `Helsinki-NLP/opus-mt-en-de`. Model ini adalah bagian dari proyek Helsinki-NLP dan dirancang untuk menerjemahkan teks dari bahasa Inggris (`en`) ke bahasa Jerman (`de`).

---
# Kesimpulan
Framework Hugging Face Transformers memberikan solusi yang efektif dan fleksibel untuk berbagai tugas NLP. Pipeline yang disediakan tidak hanya mempermudah implementasi, tetapi juga mendukung model pra-latih berkualitas tinggi untuk menghasilkan output yang akurat. Meskipun demikian, ada keterbatasan dalam menangani nuansa bahasa yang kompleks sehingga pemahaman konteks tetap perlu diperhatikan. Dengan memanfaatkan perangkat keras yang memadai dan model yang sesuai, Hugging Face Transformers menjadi alat yang sangat efektif untuk pemrosesan bahasa alami dalam berbagai aplikasi.


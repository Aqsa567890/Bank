{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP7pjwUyGCFYYMvFX60eEVE",
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Aqsa567890/Bank/blob/main/fake_and_true_news_detection.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "10Fzlr43dkie",
        "outputId": "e36bd263-ea39-4cf9-c958-993a7aaf0cc2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: nltk in /usr/local/lib/python3.11/dist-packages (3.9.1)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.11/dist-packages (from nltk) (8.1.8)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.11/dist-packages (from nltk) (1.4.2)\n",
            "Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.11/dist-packages (from nltk) (2024.11.6)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from nltk) (4.67.1)\n"
          ]
        }
      ],
      "source": [
        "pip install nltk"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "fake = pd.read_csv('Fake.csv')\n",
        "real = pd.read_csv('True.csv')\n",
        "\n",
        "# Label the datasets\n",
        "fake['label'] = 0\n",
        "real['label'] = 1\n",
        "\n",
        "# Combine datasets\n",
        "data = pd.concat([fake, real])\n",
        "data = data.sample(frac=1).reset_index(drop=True)  # Shuffle"
      ],
      "metadata": {
        "id": "cypG21lud6AZ"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "\n",
        "nltk.download('stopwords')\n",
        "nltk.download('wordnet')\n",
        "\n",
        "stop_words = set(stopwords.words('english'))\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "def preprocess_text(text):\n",
        "    text = re.sub(r'http\\S+|www\\S+|[^a-zA-Z]', ' ', text)  # remove URLs and non-letters\n",
        "    text = text.lower()\n",
        "    tokens = text.split()\n",
        "    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]\n",
        "    return ' '.join(tokens)\n",
        "\n",
        "data['text'] = data['title'] + \" \" + data['text']\n",
        "data['text'] = data['text'].apply(preprocess_text)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V8-r4GTge6ya",
        "outputId": "6964dd3c-2c49-4274-e3da-a3c51ddae0fc"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "\n",
        "nltk.download('stopwords')\n",
        "nltk.download('wordnet')\n",
        "\n",
        "stop_words = set(stopwords.words('english'))\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "def preprocess_text(text):\n",
        "    text = re.sub(r'http\\S+|www\\S+|[^a-zA-Z]', ' ', text)  # remove URLs and non-letters\n",
        "    text = text.lower()\n",
        "    tokens = text.split()\n",
        "    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]\n",
        "    return ' '.join(tokens)\n",
        "\n",
        "data['text'] = data['title'] + \" \" + data['text']\n",
        "data['text'] = data['text'].apply(preprocess_text)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KHAvFmtXfXIs",
        "outputId": "6a05664e-94ce-4c80-e348-96d1c4c52e59"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Package wordnet is already up-to-date!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "\n",
        "vectorizer = TfidfVectorizer(max_features=5000)\n",
        "X = vectorizer.fit_transform(data['text'])\n",
        "y = data['label']"
      ],
      "metadata": {
        "id": "xwLQJEm_fTJ6"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import classification_report\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n",
        "\n",
        "model = MultinomialNB()\n",
        "model.fit(X_train, y_train)\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "print(classification_report(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "i54T-x03fedn",
        "outputId": "4ab4626c-2830-464d-86ba-18c6446a8aa8"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.93      0.94      0.94      4694\n",
            "           1       0.93      0.93      0.93      4286\n",
            "\n",
            "    accuracy                           0.93      8980\n",
            "   macro avg       0.93      0.93      0.93      8980\n",
            "weighted avg       0.93      0.93      0.93      8980\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle\n",
        "\n",
        "pickle.dump(model, open('fake_news_model.pkl', 'wb'))\n",
        "pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))\n"
      ],
      "metadata": {
        "id": "tJwHB85PlgJ2"
      },
      "execution_count": 7,
      "outputs": []
    }
  ]
}

import os
import string
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


# Превращает текст письма в множество слов
def get_words(text):
    text = text.lower()

    # Убираем пунктуацию
    for symbol in string.punctuation:
        text = text.replace(symbol, " ")

    # set нужен, чтобы одинаковое слово
    # в одном письме учитывалось только один раз
    return set(text.split())


# Читает все письма из папки
def read_emails(folder, label):
    emails = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if os.path.isfile(path):
            with open(path, "r", encoding="latin-1", errors="ignore") as file:
                text = file.read()

            emails.append((text, label))

    return emails


# Читаем письма Enron1
spam_emails = read_emails("enron1/spam", "spam")
ham_emails = read_emails("enron1/ham", "ham")

emails = spam_emails + ham_emails


# Делим данные: 80% на обучение, 20% на проверку
train_emails, test_emails = train_test_split(
    emails,
    test_size=0.2,
    random_state=42,
    stratify=[label for text, label in emails]
)


# Сколько обучающих писем каждого класса
spam_count = 0
ham_count = 0

# Сколько spam/ham писем содержат каждое слово
spam_words = {}
ham_words = {}


# Обучение
for text, label in train_emails:
    words = get_words(text)

    if label == "spam":
        spam_count += 1

        for word in words:
            spam_words[word] = spam_words.get(word, 0) + 1

    else:
        ham_count += 1

        for word in words:
            ham_words[word] = ham_words.get(word, 0) + 1


# Классифицирует письмо как spam или ham
def classify_email(text):
    words = get_words(text)

    total_count = spam_count + ham_count

    # P(spam) и P(ham)
    spam_result = math.log(spam_count / total_count)
    ham_result = math.log(ham_count / total_count)

    for word in words:
        # Сколько писем каждого класса содержали слово
        word_in_spam = spam_words.get(word, 0)
        word_in_ham = ham_words.get(word, 0)

        # Сглаживание Лапласа
        p_word_spam = (word_in_spam + 1) / (spam_count + 2)
        p_word_ham = (word_in_ham + 1) / (ham_count + 2)

        # Вместо произведения вероятностей
        # складываем их логарифмы
        spam_result += math.log(p_word_spam)
        ham_result += math.log(p_word_ham)

    if spam_result > ham_result:
        return "spam"

    return "ham"


# Проверяем модель на тестовых 20%
real_labels = []
predicted_labels = []

for text, label in test_emails:
    real_labels.append(label)
    predicted_labels.append(classify_email(text))


# Метрики из задания
accuracy = accuracy_score(real_labels, predicted_labels)

sensitivity = recall_score(
    real_labels,
    predicted_labels,
    pos_label="spam"
)

specificity = recall_score(
    real_labels,
    predicted_labels,
    pos_label="ham"
)


print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
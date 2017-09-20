# YandexMeteumSolution
1st place solution for Yandex.Meteum hackathon


1) Отдельно для каждого города генерим фичи:

Файлик compute.py генерит одну группу фичей для тренировочных данных
compute_test.py такую же группу для тестовых данных

Аналогично, radius_compute.py и radius_compute_test.py

2) Запускаем train.py

Объединяем все фичи, выбрасываем те, которых нет в тестовых данных, запускаем catboost.


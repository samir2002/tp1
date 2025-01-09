import face_recognition
known_image = face_recognition.load_image_file('file1.jpg')
unknown_image = face_recognition.load_image_file('file2.jpg')
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
results = face_recognition.compare_faces([biden_encoding],
unknown_encoding)
print(str(results))     
# Entraîner le modèle pour la compatibilité avec les critères
model_compat = LogisticRegression()
model_compat.fit(X_train_scaled, y_train_compat)

# Prédictions
y_pred_compat = model_compat.predict(X_test_scaled)

# Évaluation du modèle de compatibilité
print("Compatibilité - Classification Report:")
print(classification_report(y_test_compat, y_pred_compat))
print("Compatibilité - Confusion Matrix:")
print(confusion_matrix(y_test_compat, y_pred_compat))
# Entraîner le modèle pour la disponibilité pour le mercato
model_free = LogisticRegression()
model_free.fit(X_train_scaled, y_train_free)

# Prédictions
y_pred_free = model_free.predict(X_test_scaled)

# Évaluation du modèle de disponibilité
print("Disponibilité pour le mercato - Classification Report:")
print(classification_report(y_test_free, y_pred_free))
print("Disponibilité pour le mercato - Confusion Matrix:")
print(confusion_matrix(y_test_free, y_pred_free))

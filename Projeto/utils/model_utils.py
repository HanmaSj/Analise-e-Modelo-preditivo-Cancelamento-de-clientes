from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib


def evaluate_model(model, X_test, y_test, task='classification'):
    """Avalia o modelo nos dados de teste."""
    predictions = model.predict(X_test)
    if task == 'classification':
        score = accuracy_score(y_test, predictions)
    elif task == 'regression':
        score = mean_squared_error(y_test, predictions, squared=False)  # RMSE
    else:
        raise ValueError("A tarefa deve ser 'classification' ou 'regression'.")
    return score

def save_model(model, file_path):
    """Salva o modelo treinado em um arquivo."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Carrega um modelo treinado de um arquivo."""
    return joblib.load(file_path)

def test_models(X_train, X_test, y_train, y_test):
    """Testa múltiplos modelos de classificação nos dados fornecidos e retorna suas performances."""
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = evaluate_model(model, X_test, y_test, task='classification')
        results[name] = score

    return results

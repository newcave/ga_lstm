import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from deap import base, creator, tools, algorithms

# 가상의 데이터를 생성합니다.
def generate_data():
    # 임의의 데이터 생성 (100시간 3변수)
    data = np.random.rand(100, 3)
    df = pd.DataFrame(data, columns=['var1', 'var2', 'var3'])
    return df

# 데이터 준비 함수
def prepare_data(df):
    X = df.values
    y = np.random.rand(len(df), 1)  # 임의의 타겟 값
    return X, y

# LSTM 모델을 생성하고 학습하는 함수
def create_and_train_lstm(params, X_train, y_train):
    lstm_units, learning_rate, batch_size = params

    model = Sequential()
    model.add(LSTM(int(lstm_units), input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=int(batch_size), verbose=0)
    loss = model.evaluate(X_train, y_train, verbose=0)
    
    return loss

# 평가 함수
def evaluate(params):
    X, y = prepare_data(generate_data())
    X = X.reshape((X.shape[0], X.shape[1], 1))
    loss = create_and_train_lstm(params, X, y)
    return (loss,)

# 유전 알고리즘 설정
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 10, 100)  # LSTM 유닛 수
toolbox.register("attr_lr", np.random.uniform, 0.0001, 0.01)  # 학습률
toolbox.register("attr_batch", np.random.randint, 16, 128)  # 배치 크기

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_float, toolbox.attr_lr, toolbox.attr_batch), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[10, 0.0001, 16], up=[100, 0.01, 128], eta=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    st.title("LSTM Hyperparameter Optimization using GA")

    population_size = st.sidebar.slider("Population Size", 5, 20, 10)
    generations = st.sidebar.slider("Generations", 1, 10, 5)
    crossover_prob = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.5)
    mutation_prob = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.2)

    if st.button("Run Optimization"):
        population = toolbox.population(n=population_size)

        for gen in range(generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
            fits = map(toolbox.evaluate, offspring)
            
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            population = toolbox.select(offspring, k=len(population))
        
        top1 = tools.selBest(population, k=1)[0]
        st.write("최적의 하이퍼파라미터:", top1)
        st.write("최적의 손실 값:", top1.fitness.values[0])

if __name__ == "__main__":
    main()

# [ES] Clasificación Binaria de Imágenes Industriales: Optimización de Modelos CNN

Este repositorio documenta la refactorización técnica de un proyecto académico orientado a la clasificación de herramientas de trabajo (**Martillos vs. Destornilladores**).

El objetivo de esta versión (2026) fue migrar una implementación base ineficiente hacia un pipeline moderno utilizando **TensorFlow** y **Keras**, resolviendo problemas de gestión de memoria y sobreajuste (*overfitting*).

## Contexto de la Refactorización

El proyecto original (2023) utilizaba métodos manuales de carga de datos que saturaban la memoria RAM y modelos que no generalizaban correctamente. En esta actualización se implementaron las siguientes mejoras técnicas:

### 1. Optimización del Pipeline de Datos (ETL)
* **Enfoque Tradicional:** Carga de imágenes en listas de Python y normalización manual en CPU.
* **Implementación Actual:** Uso de `tf.data.Dataset` con `cache()` y `prefetch(AUTOTUNE)`. Esto habilita la carga perezosa (*lazy loading*) y paraleliza el procesamiento entre CPU y GPU para evitar cuellos de botella.

### 2. Estrategia de Modelado (End-to-End)
Se integró el preprocesamiento dentro de la arquitectura del modelo. Las capas de **Rescaling** y **Data Augmentation** ahora forman parte del grafo computacional, aprovechando la aceleración de hardware de la GPU.

## Evolución del Modelo y Benchmarking

Se realizó un entrenamiento comparativo iterativo entre cuatro arquitecturas para demostrar la evolución del rendimiento y justificar las decisiones de diseño:

| Iteración | Arquitectura | Accuracy (Val) | Análisis Técnico |
| :--- | :--- | :--- | :--- |
| **1. Baseline** | Dense (MLP) | ~58% | **Underfitting.** Confirmó que el aplanamiento (*Flatten*) de la entrada destruye la información espacial necesaria para distinguir formas complejas. |
| **2. CNN Básica** | Conv2D + MaxPool | 60% | **High Variance (Overfitting).** El modelo alcanzó 99% en entrenamiento pero falló en validación, memorizando el ruido del dataset. |
| **3. CNN + Dropout** | Conv2D + Dropout (0.5) | ~65% | La regularización ayudó levemente, pero no fue suficiente para contrarrestar la escasez de datos. |
| **4. Final (SOTA)** | CNN + Augmentation | **72%** | **Estable.** La incorporación de ruido controlado (*RandomFlip, Rotation, Zoom*) estabilizó las curvas de aprendizaje y logró generalización real. |

## Evaluación de Resultados

El modelo final fue evaluado con un set de validación de 669 imágenes.
* **Métrica Final:** 72% de Precisión Global.
* **Diagnóstico de Error:** La matriz de confusión indica que el modelo identifica mejor los **Destornilladores** (Recall 0.75), mientras que presenta mayor confusión con los **Martillos** (Recall 0.70) en ángulos donde la cabeza de la herramienta no es prominente.

## Stack Tecnológico

* **Framework:** TensorFlow 2.x / Keras.
* **Procesamiento:** OpenCV, NumPy.
* **Visualización:** Matplotlib, Seaborn, TensorBoard.
* **Exportación:** TensorFlow.js (Conversión del modelo para inferencia en cliente).

## Trabajo Futuro

Dado que el rendimiento actual está limitado principalmente por el tamaño del dataset (~600 imágenes por clase), la hoja de ruta técnica incluye:

1.  **Transfer Learning:** Implementación de arquitecturas pre-entrenadas (MobileNetV2 o ResNet50) para aprovechar características visuales aprendidas y superar el 90% de precisión.
2.  **Expansión del Dataset:** Recolección de nuevas muestras con variaciones de iluminación y fondo para robustecer la clase "Martillo".

---
**Autora:** Agustina Delgado

---

# [EN] Industrial Image Binary Classification: CNN Optimization

This repository documents the technical refactoring of an academic project aimed at classifying workshop tools (**Hammers vs. Screwdrivers**).

The goal of this version (2026) was to migrate an inefficient baseline implementation towards a modern pipeline using **TensorFlow** and **Keras**, solving critical memory management and overfitting issues.

## Refactoring Context

The original project (2023) relied on manual data loading methods that saturated RAM and produced models capable of learning but unable to generalize. This update implements the following technical improvements:

### 1. Data Pipeline Optimization (ETL)
* **Legacy Approach:** Loading images into native Python lists and performing manual normalization on the CPU.
* **Current Implementation:** Utilization of `tf.data.Dataset` with `cache()` and `prefetch(AUTOTUNE)`. This enables lazy loading and parallelizes processing between CPU and GPU to prevent bottlenecks.

### 2. Modeling Strategy (End-to-End)
Preprocessing was integrated directly into the model architecture. **Rescaling** and **Data Augmentation** layers are now part of the computational graph, leveraging GPU hardware acceleration.

## Model Evolution and Benchmarking

An iterative comparative training process was conducted across four architectures to demonstrate performance evolution and justify design decisions:

| Iteration | Architecture | Accuracy (Val) | Technical Analysis |
| :--- | :--- | :--- | :--- |
| **1. Baseline** | Dense (MLP) | ~58% | **Underfitting.** Confirmed that Flattening the input destroys the spatial information required to distinguish complex shapes. |
| **2. Basic CNN** | Conv2D + MaxPool | 60% | **High Variance (Overfitting).** The model reached 99% on training data but failed on validation, effectively memorizing dataset noise. |
| **3. CNN + Dropout** | Conv2D + Dropout (0.5) | ~65% | Regularization provided a slight improvement but was insufficient to counteract the data scarcity. |
| **4. Final (SOTA)** | CNN + Augmentation | **72%** | **Stable.** The incorporation of controlled noise (*RandomFlip, Rotation, Zoom*) stabilized learning curves and achieved true generalization. |

## Performance Evaluation

The final model was evaluated using a validation set of 669 images.
* **Final Metric:** 72% Global Accuracy.
* **Error Diagnosis:** The confusion matrix indicates the model is more effective at identifying **Screwdrivers** (Recall 0.75), while showing higher confusion with **Hammers** (Recall 0.70), particularly in angles where the tool head is not prominent.

## Tech Stack

* **Framework:** TensorFlow 2.x / Keras.
* **Processing:** OpenCV, NumPy.
* **Visualization:** Matplotlib, Seaborn, TensorBoard.
* **Export:** TensorFlow.js (Model conversion for client-side inference).

## Future Work

Given that current performance is primarily limited by dataset size (~600 images per class), the technical roadmap includes:

1.  **Transfer Learning:** Implementation of pre-trained architectures (MobileNetV2 or ResNet50) to leverage learned visual features and surpass 90% accuracy.
2.  **Dataset Expansion:** Collection of new samples with lighting and background variations to improve robustness for the "Hammer" class.

---
**Author:** Agustina Delgado

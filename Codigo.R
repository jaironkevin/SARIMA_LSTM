############################ Librerías necesarias #############################
# Importación y manejo de datos
library(readxl)         
library(dplyr)          
library(lubridate)      
library(tsibble)        
library(tidyr)
library(scales)

# Análisis exploratorio y visualización
library(ggplot2)        
library(forecast)       
library(e1071)          
library(magrittr)       
library(ggfortify)      

# Modelado y ajuste de modelos SARIMA
library(fable)
library(forecast)
library(fabletools)     
library(kableExtra)     
library(Metrics)        


# Modelado LSTM
library(tensorflow) 
library(keras3)   




# Extensión de código para ordenar resultados






# IMPORTACIÓN DE DATOS
AGRO <- readxl::read_excel("D:/A_TESIS_ESTADISTICA/CAP_RESULTADOS/DATASET/AGRO.xlsx")


################################################################################
####################### O.E.1: DESCRIPCIÓN DE LA SERIE #########################
################################################################################

# Defino las medidas a calcular
`%>%` <- magrittr::`%>%`
Medidas <- c("obs.", "Mínimo", "1st qu.", "Mediana", "Media", "3rd Qu.", 
             "Máximo", "SD.","Asimetria", "Curtosis", "Moda")


# Calculo de las estadísticas en AGRO$Producción
Z <- vector(length = length(Medidas))
Z[1] <- round(sum(!is.na(AGRO$Producción)), 0)
Z[2] <- round(min(AGRO$Producción, na.rm = TRUE), 0)
Z[3] <- round(quantile(AGRO$Producción, 0.25, na.rm = TRUE), 0)
Z[4] <- round(median(AGRO$Producción, na.rm = TRUE), 0)
Z[5] <- round(mean(AGRO$Producción, na.rm = TRUE), 0)
Z[6] <- round(quantile(AGRO$Producción, 0.75, na.rm = TRUE), 0)
Z[7] <- round(max(AGRO$Producción, na.rm = TRUE), 0)
Z[8] <- round(sd(AGRO$Producción, na.rm = TRUE), 3)
Z[9] <- round(e1071::skewness(AGRO$Producción, na.rm = TRUE), 3)
Z[10] <- round(e1071::kurtosis(AGRO$Producción, na.rm = TRUE), 3)
Z[11] <- round(Moda(AGRO$Producción), 3)

# Crear el data frame
Z <- data.frame("Estadística" = Medidas, Valor = Z)


# Grafica
Mango <- ts(AGRO$Producción, start = c(2000, 1), end = c(2024, 8), frequency = 12)

autoplot(Mango, colour = "#AE123A")+ 
  xlab("Tiempo") +
  ylab("Producción (toneladas)")

# Veo la estacionalidad
ggseasonplot(Mango) +
  labs(
    title = NULL,
    x = NULL,
    y = "Producción",
    caption = "Meses"
  ) +
  theme(
    plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    plot.caption = element_text(size = 12, hjust = 0.5)
  ) +
  scale_fill_brewer(palette = "Dark2") +
  geom_point(data = NULL)



fechas <- seq(as.Date("2000-01-01"), as.Date("2024-08-01"), by = "month")

Box_mensual <- data.frame(
  Fecha = fechas,
  Produccion = AGRO$Producción 
)
df_mango<-Box_mensual 
Box_mensual$Mes <- factor(month.name[month(Box_mensual$Fecha)], levels = month.name)

ggplot(Box_mensual, aes(x = Mes, y = Produccion, fill = Mes)) +
  geom_boxplot(alpha = 0.7) +  
  geom_jitter(position = position_jitter(0.2), color = "black", alpha = 0.5) +  
  labs(x = "Mes",
       y = "Producción (toneladas)") +
  scale_fill_brewer(palette = "Set3") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 


# reviso los conponentes que tiene la serie
componetes <- stl(Mango, s.window = "periodic")
autoplot(componetes, facets = TRUE, ts.colour = "#CD5555")

############################# Estacionariedad ##################################

adf_table(Mango)
kpss_table(Mango)


# recordando que no se recomienda realizar 3 o más operaciones de diferenciación
#D=1
Diff_Mango <- diff(Mango, lag = 12)


p1 <- autoplot(Diff_Mango, colour = "#26456E") +
  theme(plot.title = element_text(hjust = 0.5, size = 10)) 

p1


################################################################################
########################## O.E.2: SARIMA-BOX-JENKINS ###########################
################################################################################


# Identificación SARIMA
D1<-afc.id(Diff_Mango, lag.max = 25, period = 12, threshold = 0.05)
D2<-pacf.id(Diff_Mango, lag.max = 25, period = 12, threshold = 0.05)


# Estimación SARIMA

models_sarima <- AGRO %>%
  mutate(Date = yearmonth(Date)) %>%
  as_tsibble(index = Date)

models_sarima[,-1] <- apply(models_sarima[,-1], 2, as.numeric)


MODELS_SARIMA_G1 <- models_sarima %>% 
  model(
    G1_MOD1 = ARIMA(Producción ~ pdq(1,0,1) + PDQ(2,1,0)),
    G1_MOD2 = ARIMA(Producción ~ pdq(1,0,1) + PDQ(2,1,1)),
    G1_MOD3 = ARIMA(Producción ~ pdq(1,0,0) + PDQ(2,1,0)),
    G1_MOD4 = ARIMA(Producción ~ pdq(0,0,1) + PDQ(2,1,0)),
    G1_MOD5 = ARIMA(Producción ~ pdq(1,0,0) + PDQ(0,1,1)),
    G1_MOD6 = ARIMA(Producción ~ pdq(0,0,1) + PDQ(2,1,1)),
    G1_MOD7 = ARIMA(Producción ~ pdq(0,0,1) + PDQ(0,1,1))
  ) %>%
  tidy() %>%
  select(.model, term, estimate, std.error, statistic, p.value) %>%
  mutate(
    estimate = sprintf("%.3f", estimate),
    std.error = sprintf("%.3f", std.error),
    statistic = sprintf("%.3f", statistic),
    p.value = sprintf("%.3f", p.value),
    .model = recode(.model, 
                    G1_MOD1 = "SARIMA(1,0,1)(2,1,0)[12]", 
                    G1_MOD2 = "SARIMA(1,0,1)(2,1,1)[12]",
                    G1_MOD3 = "SARIMA(1,0,0)(2,1,0)[12]",
                    G1_MOD4 = "SARIMA(0,0,1)(2,1,0)[12]",
                    G1_MOD5 = "SARIMA(1,0,0)(0,1,1)[12]",
                    G1_MOD6 = "SARIMA(0,0,1)(2,1,1)[12]",
                    G1_MOD7 = "SARIMA(0,0,1)(0,1,1)[12]")
  )



# Diagnostico SARIMA

MOD3<-Arima(Mango, order=c(1,0,0), seasonal=c(2,1,0),method = "ML")
check_residuals(MOD3, lag.max = 25, title = "SARIMA (1,0,0)(2,1,0)[12]")

MOD3_test <- test.residuals(MOD3, 
                            test_ac = "Ljung-Box",
                            test_normal = "Jarque-Bera", 
                            test_hetero = "Levene", 
                            test_stationarity = "KPSS",
                            test_media = TRUE)



MOD4<-Arima(Mango, order=c(0,0,1), seasonal=c(2,1,0), method = "ML")
check_residuals(MOD4, lag.max = 25, title = "SARIMA(0,0,1)(2,1,0)[12]")

MOD4_test <- test.residuals(MOD4, 
                            test_ac = "Ljung-Box",
                            test_normal = "Jarque-Bera", 
                            test_hetero = "Levene", 
                            test_stationarity = "KPSS",
                            test_media = TRUE)



MOD5<-Arima(Mango, order=c(1,0,0), seasonal=c(0,1,1), method = "ML")
check_residuals(MOD5, lag.max = 25, title = "SARIMA(1,0,0)(0,1,1)[12]")


MOD5_test <- test.residuals(MOD5, 
                            test_ac = "Ljung-Box",
                            test_normal = "Jarque-Bera", 
                            test_hetero = "Levene", 
                            test_stationarity = "KPSS",
                            test_media = TRUE)




MOD7 <- Arima(Mango, order = c(0,0,1), seasonal = list(order = c(0,1,1), period = 12), method = "ML")
check_residuals(MOD7, lag.max = 25, title = "SARIMA(0,0,1)(0,1,1)[12]")

MOD7_test <- test.residuals(MOD7, 
                            test_ac = "Ljung-Box",
                            test_normal = "Jarque-Bera", 
                            test_hetero = "Levene", 
                            test_stationarity = "KPSS",
                            test_media = TRUE)


# Comparación entre modelos los 4 SARIMA

# Preparo el conjunto de datos
models_sarima <- AGRO %>%
  mutate(Date = yearmonth(Date)) %>%
  as_tsibble(index = Date)

models_sarima[,-1] <- apply(models_sarima[,-1], 2, as.numeric)

# Definir los modelos SARIMA usando los órdenes ARIMA como nombres de modelo
models_sarima_fable <- models_sarima %>% 
  model(
    `ARIMA(1,0,0)(2,1,0)` = ARIMA(Producción ~ pdq(1,0,0) + PDQ(2,1,0)),
    `ARIMA(0,0,1)(2,1,0)` = ARIMA(Producción ~ pdq(0,0,1) + PDQ(2,1,0)),
    `ARIMA(1,0,0)(0,1,1)` = ARIMA(Producción ~ pdq(1,0,0) + PDQ(0,1,1)),
    `ARIMA(0,0,1)(0,1,1)` = ARIMA(Producción ~ pdq(0,0,1) + PDQ(0,1,1))
  )


# Obtengo los ajustes
ajuste <- fitted(models_sarima_fable)
ajuste_limpio <- ajuste %>%
  filter(!is.na(.fitted)) 

# Gráfico: Todos los modelos en un solo gráfico
plot_all <- ggplot(models_sarima, aes(x = Date, y = Producción)) +
  geom_line(aes(color = "Observado"), linewidth = 0.3) +  
  geom_line(data = ajuste_limpio, aes(y = .fitted, color = .model), 
            linewidth = 0.5) +  
  labs(color = "Modelos") +
  theme_minimal() +
  scale_color_manual(values = c("Observado" = "black", 
                                "ARIMA(1,0,0)(2,1,0)" = "#1f77b4",
                                "ARIMA(0,0,1)(2,1,0)" = "#ff7f0e",
                                "ARIMA(1,0,0)(0,1,1)" = "#2ca02c",
                                "ARIMA(0,0,1)(0,1,1)" = "#d62728")) 

plot_all


######################## Métricas entre modelos SARIMA #########################


# Preparar el conjunto de datos como tsibble
models_sarima <- AGRO %>%
  mutate(Date = yearmonth(Date)) %>%
  as_tsibble(index = Date)

models_sarima[,-1] <- apply(models_sarima[,-1], 2, as.numeric)

# Ajustar los modelos SARIMA
models_sarima_fable <- models_sarima %>% 
  model(
    `ARIMA(1,0,0)(2,1,0)` = ARIMA(Producción ~ pdq(1,0,0) + PDQ(2,1,0)),
    `ARIMA(0,0,1)(2,1,0)` = ARIMA(Producción ~ pdq(0,0,1) + PDQ(2,1,0)),
    `ARIMA(1,0,0)(0,1,1)` = ARIMA(Producción ~ pdq(1,0,0) + PDQ(0,1,1)),
    `ARIMA(0,0,1)(0,1,1)` = ARIMA(Producción ~ pdq(0,0,1) + PDQ(0,1,1))
  )

# Extraer los valores observados y predichos 
augment_data <- models_sarima_fable %>%
  augment() %>%  
  select(.model, Producción, .fitted)  


custom_metrics <- augment_data %>%
  as.data.frame() %>%  
  group_by(.model) %>% 
  summarise(
    MDAE = mdae(Producción, .fitted),  
    SMAPE = smape(Producción, .fitted) 
  )

custom_metrics


# Calcular las métricas estándar de fable
accuracy_df <- models_sarima_fable %>%
  forecast::accuracy()

# Combinar las métricas estándar con las de metrics 
full_metrics_df <- accuracy_df %>%
  left_join(custom_metrics, by = ".model")  

full_metrics_df

################################################################################
############################# O.E.3: Modelado LSTM #############################
################################################################################
# Nota: Para no extender mas el código, se presenta el código del cual se fueron 
# modificando los hiperparametros y parámetros de los modelos A, B, C, D.

# Vuelvo a Carga los datos disponibles (Fuente: MIDAGRI)
file_path <- "D:/A_TESIS_ESTADISTICA/CAP_RESULTADOS/DATASET/AGRO.xlsx"
df <- read_excel(file_path)
df$Date <- as.Date(paste0(df$Date, "-01"), format = "%Y-%m-%d")
df <- df[order(df$Date), ]
str(df)

############################### 1.Pre-procesamiento ##############################

# 1.1. Partición del set en entrenamiento, validación y prueba
splits <- train_val_test_split(df$Producción, tr_size = 0.7, vl_size = 0.15, ts_size = 0.15)
tr <- splits$train
vl <- splits$val
ts <- splits$test


cat(sprintf('Tamaño set de entrenamiento: %d\n', length(tr)))
cat(sprintf('Tamaño set de validación: %d\n', length(vl)))
cat(sprintf('Tamaño set de prueba: %d\n', length(ts)))

# 1.2. Generación del dataset supervisado (entrada y salida del modelo)

# Definir hiperparámetros
INPUT_LENGTH <- 24  # Número de pasos temporales usados como entrada
OUTPUT_LENGTH <- 17  # Número de pasos que queremos predecir o de salida

datasets <- crear_todos_los_datasets(tr, vl, ts, INPUT_LENGTH, OUTPUT_LENGTH)

x_tr <- datasets$x_tr
y_tr <- datasets$y_tr
x_vl <- datasets$x_vl
y_vl <- datasets$y_vl
x_ts <- datasets$x_ts
y_ts <- datasets$y_ts


# Preparar la lista de datos de entrada
data_in <- list(
  'x_tr' = x_tr, 'y_tr' = y_tr,
  'x_vl' = x_vl, 'y_vl' = y_vl,
  'x_ts' = x_ts, 'y_ts' = y_ts
)

# 1.3. Escalamiento
resultado <- escalar_dataset(data_in)
data_s <- resultado$data_scaled
scalers <- resultado$scalers

# Extraer los datasets escalados
x_tr_s <- data_s$x_tr_s
y_tr_s <- data_s$y_tr_s
x_vl_s <- data_s$x_vl_s
y_vl_s <- data_s$y_vl_s
x_ts_s <- data_s$x_ts_s
y_ts_s <- data_s$y_ts_s

# Ajustar las formas de las salidas (Y) para que coincidan con la salida del modelo
y_tr_s <- array(y_tr_s, dim = c(dim(y_tr_s)[1], dim(y_tr_s)[2]))
y_vl_s <- array(y_vl_s, dim = c(dim(y_vl_s)[1], dim(y_vl_s)[2]))
y_ts_s <- array(y_ts_s, dim = c(dim(y_ts_s)[1], dim(y_ts_s)[2]))


##################### 2. Creación y entrenamiento del modelo ###################

# Establecer una semilla para reproducibilidad
set.seed(123)
tensorflow::set_random_seed(123)


# 2.1. Definir hiperparámetros
INPUT_LENGTH <- 24   # Número de pasos temporales usados como entrada  12
OUTPUT_LENGTH <- 17  # Número de pasos que queremos predecir
N_UNITS_FIRST_LSTM <- 256 #128
N_UNITS_SECOND_LSTM <- 128 # 64
DROPOUT_RATE <- 0.2
EPOCHS <- 50
BATCH_SIZE <- 32


# 2.2. Modelado
# Definir la forma de la entrada
INPUT_SHAPE <- c(INPUT_LENGTH, dim(x_tr_s)[3])  


modelo <- keras_model_sequential() %>%
  layer_lstm(units = N_UNITS_FIRST_LSTM, return_sequences = TRUE, input_shape = INPUT_SHAPE) %>%
  layer_dropout(rate = DROPOUT_RATE) %>%
  layer_lstm(units = N_UNITS_SECOND_LSTM) %>%
  layer_dense(units = OUTPUT_LENGTH, activation = 'linear')



# Compilar el modelo
modelo %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "mae"
)

# revisamos el esquema del modelo
summary(modelo)
plot(modelo)
plot( 
  modelo, 
  show_shapes = TRUE, 
  show_dtype = TRUE, 
  show_layer_names = TRUE, 
  rankdir = "TB", 
  expand_nested = TRUE, 
  dpi = 100, 
  show_layer_activations = TRUE, 
) 


# Definir EarlyStopping
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  restore_best_weights = TRUE
)

# Entrenar el modelo
historiaA <- modelo %>% fit(
  x = x_tr_s,
  y = y_tr_s,
  batch_size = BATCH_SIZE,
  epochs = EPOCHS,
  validation_data = list(x_vl_s, y_vl_s),
  verbose = 2,
  callbacks = list(early_stop)
)

# ploteamos la historia del entrenamiento y validación
plotC<-plot_history(historiaA) + labs(title = "Modelo A: inputs=24, outputs=17, 2 capas (256, 128 units).") +
  theme(plot.title = element_text(hjust = 0.5, size = 10, face = "bold"))



# guardo y cargo el historial de entrenamiento
save(historia, file ="history1.RData")
load("history1.RData")
# Guardar el modelo completo en formato .keras
keras3::save_model(modelo,"LSTM_A.keras")

######################## 3. Desempeño del Modelo ###############################

# Evaluar el modelo en los 3 conjuntos 

mae_tr <- modelo %>% evaluate(x_tr_s, y_tr_s, verbose = 0)
mae_vl <- modelo %>% evaluate(x_vl_s, y_vl_s, verbose = 0)
mae_ts <- modelo %>% evaluate(x_ts_s, y_ts_s, verbose = 0)

cat('Comparativo desempeños:\n')
cat(sprintf('  MAE train:\t %.3f\n', mae_tr$loss))
cat(sprintf('  MAE val:\t %.3f\n', mae_vl$loss))
cat(sprintf('  MAE test:\t %.3f\n', mae_ts$loss))


####################### 4. Evaluación del modelo sobre Test ####################


# 4.1. Hacer predicciones en el conjunto de prueba
y_ts_pred_s <- modelo %>% predict(x_ts_s, verbose = 0)

# 4.2. Desescalar las predicciones y los valores observados
y_ts_pred <- desescalar(y_ts_pred_s, scalers$scaler_y)
y_ts_true <- desescalar(y_ts_s, scalers$scaler_y)

# 4.2. Graficos las secuencias que se generaran y pronósticos finales
# Debido al solapamiento de los outputs (many to many), se generan mas de un 
# pronostico para cada punto de test observado.


data <- resultLSTM_testC(y_ts_pred, y_ts_true)
plot_resultLSTM_testC(data$test_completo, data$resultados)


data_completa <- resultLSTM_complete(y_ts_pred, y_ts_true)
plot_resultLSTM_complete(data_completa$full_data, data_completa$test_completo, data_completa$resultados)

# 4.2. se aplican medidas de precisión basadas en sus errores

# se consideran las mas robustas debido a la naturaleza de la serie 
Test_obs<-tibble_pre_obs(y_ts_true, y_ts_pred, df)
metrics(Test_obs$Observado, Test_obs$Predicción)


################################################################################
############################# O.E.4: LSTM VS SARIMA ############################
################################################################################


# Dado que ya se entreno y valido el LSTM A, solo se tomara en cuenta los
# estadísticos de ajuste de ese modelo del test y se vuelve a correr el mejor 
# modelo SARIMA pero tomando en cuenta la cantidad exacta del test LSTM para que 
# sea comparable. 


# métricas o estadísticos de ajuste de LSTM A en el test.
metrics(Test_obs$Observado, Test_obs$Predicción)


# métricas o estadísticos de ajuste de SARIMA(1,0,0)(2,1,0) en el test.
AGRO_tsibble <- AGRO %>%
  mutate(Date = yearmonth(Date)) %>%
  as_tsibble(index = Date)


AGRO_tsibble[,-1] <- apply(AGRO_tsibble[,-1], 2, as.numeric)

AGRO_tsibble %>% 
  filter(Date <= yearmonth("2022-11")) %>% 
  model(
    sarima = ARIMA(Producción ~ 0+pdq(1,0,0) + PDQ(2,1,0))
  ) %>% 
  forecast(h = 21) %>% 
  accuracy(AGRO_tsibble)


################################################################################
########################### O.E.5: Pronostico SARIMA ###########################
################################################################################

# Convertir los datos en un tsibble
Mango_predic_tsibble <- AGRO %>%
  mutate(Date = yearmonth(Date)) %>%
  as_tsibble(index = Date)

Mango_predic_tsibble[,-1] <- apply(Mango_predic_tsibble[,-1], 2, as.numeric)

Mango_predic_tsibble %>% 
  autoplot(Producción)


fit_Mango_predic <- Mango_predic_tsibble %>% 
  model(
    sarima = ARIMA(Producción ~ 0 + pdq(1,0,0) + PDQ(2,1,0))
  )


# USO EL BOOSTRAP POR NO CUMPLIRSE EL SUPUESTO DE NORMALIDAD
fc <- fit_Mango_predic %>% forecast(h = 17, bootstrap = TRUE)
fc

fc %>% 
  autoplot(Mango_predic_tsibble) +
  labs(
    y = "Producción (t)")





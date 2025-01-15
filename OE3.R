

############################### 1.Pre-procesamiento ##############################
# Función para divición de datos
train_val_test_split <- function(serie, tr_size = 0.7, vl_size = 0.15, ts_size = 0.15) {
  # Divide una serie de tiempo en conjuntos de entrenamiento, validación y prueba.
  #
  # Parámetros:
  # - serie: Serie de tiempo a dividir.
  # - tr_size: Proporción del conjunto de entrenamiento (por defecto 0.7).
  # - vl_size: Proporción del conjunto de validación (por defecto 0.15).
  # - ts_size: Proporción del conjunto de prueba (por defecto 0.15).
  #
  # Retorna:
  # - train: Subserie de entrenamiento.
  # - val: Subserie de validación.
  # - test: Subserie de prueba.
  
  # Verificar que las proporciones sumen 1
  total_size <- tr_size + vl_size + ts_size
  if (total_size != 1.0) {
    stop(sprintf("Los porcentajes deben sumar 1. El total actual es %f", total_size))
  }
  
  N <- length(serie)
  Ntrain <- floor(tr_size * N)
  Nval <- floor(vl_size * N)
  
  # Ajuste para asegurar que se utilicen todos los datos
  Ntest <- N - Ntrain - Nval
  
  train <- serie[1:Ntrain]
  val <- serie[(Ntrain + 1):(Ntrain + Nval)]
  test <- serie[(Ntrain + Nval + 1):N]
  
  return(list(train = train, val = val, test = test))
}


# función de creación de dataset para la lstm
crear_dataset_supervisado <- function(array, input_length, output_length) {
  # Crea un conjunto de datos supervisado para una red LSTM univariada multi-step.
  #
  # Parámetros:
  # - array: vector o matriz de tamaño N (univariado) o N x 1.
  # - input_length: número de instantes de tiempo de entrada.
  # - output_length: número de instantes de tiempo a predecir.
  #
  # Retorna:
  # - X: arreglo de entradas con forma (muestras, input_length, 1).
  # - Y: arreglo de salidas con forma (muestras, output_length, 1).
  
  # Asegurar que array es un vector univariado
  if (is.matrix(array) && ncol(array) == 1) {
    array <- as.vector(array)
  } else if (!is.vector(array)) {
    stop("La serie de tiempo debe ser univariada (una sola característica).")
  }
  
  total_length <- input_length + output_length
  N <- length(array)
  N_samples <- N - total_length + 1
  
  if (N_samples <= 0) {
    stop("El tamaño del array es insuficiente para los parámetros input_length y output_length dados.")
  }
  
  # Crear matriz de datos usando embed y ajustar el orden de las columnas
  data_mat <- embed(array, total_length)[, total_length:1]
  
  # Separar X y Y
  X <- data_mat[, 1:input_length]
  Y <- data_mat[, (input_length + 1):(input_length + output_length)]
  
  # Añadir dimensión extra para características
  X <- array(X, dim = c(N_samples, input_length, 1))
  Y <- array(Y, dim = c(N_samples, output_length, 1))
  
  return(list(X = X, Y = Y))
}
crear_todos_los_datasets <- function(tr, vl, ts, input_length, output_length) {
  
  
  # Asegurar que 'Producción' es numérico y vectorial
  tr_values <- as.numeric(unlist(tr))
  vl_values <- as.numeric(unlist(vl))
  ts_values <- as.numeric(unlist(ts))
  
  # Crear los datasets supervisados
  tr_result <- crear_dataset_supervisado(tr_values, input_length, output_length)
  x_tr <- tr_result$X
  y_tr <- tr_result$Y
  
  vl_result <- crear_dataset_supervisado(vl_values, input_length, output_length)
  x_vl <- vl_result$X
  y_vl <- vl_result$Y
  
  ts_result <- crear_dataset_supervisado(ts_values, input_length, output_length)
  x_ts <- ts_result$X
  y_ts <- ts_result$Y
  
  # Imprimir información sobre los tamaños
  cat('Tamaños entrada (BATCHES x INPUT_LENGTH x FEATURES) y de salida (BATCHES x OUTPUT_LENGTH x FEATURES)\n')
  cat(sprintf('Set de entrenamiento - x_tr: (%d, %d, %d), y_tr: (%d, %d, %d)\n',
              dim(x_tr)[1], dim(x_tr)[2], dim(x_tr)[3],
              dim(y_tr)[1], dim(y_tr)[2], dim(y_tr)[3]))
  cat(sprintf('Set de validación - x_vl: (%d, %d, %d), y_vl: (%d, %d, %d)\n',
              dim(x_vl)[1], dim(x_vl)[2], dim(x_vl)[3],
              dim(y_vl)[1], dim(y_vl)[2], dim(y_vl)[3]))
  cat(sprintf('Set de prueba - x_ts: (%d, %d, %d), y_ts: (%d, %d, %d)\n',
              dim(x_ts)[1], dim(x_ts)[2], dim(x_ts)[3],
              dim(y_ts)[1], dim(y_ts)[2], dim(y_ts)[3]))
  
  return(list(x_tr = x_tr, y_tr = y_tr,
              x_vl = x_vl, y_vl = y_vl,
              x_ts = x_ts, y_ts = y_ts))
}

# Escalar los datos (OJO QUE SE USA EL METODO ROBUSTO  DEBIDO A LAS CARACTERISTICAS DE LA SERIE)
escalar_dataset <- function(data_input) {
  # data_input: lista con los datasets de entrada y salida del modelo
  # (data_input = list('x_tr' = x_tr, 'y_tr' = y_tr, 'x_vl' = x_vl, 'y_vl' = y_vl,
  #                    'x_ts' = x_ts, 'y_ts' = y_ts))
  
  NFEATS <- dim(data_input$x_tr)[3]
  
  # Inicializar listas para almacenar los parámetros de escalamiento
  scalers_x <- vector("list", NFEATS)
  
  # Inicializar arreglos para almacenar los datos escalados
  x_tr_s <- data_input$x_tr
  x_vl_s <- data_input$x_vl
  x_ts_s <- data_input$x_ts
  y_tr_s <- data_input$y_tr
  y_vl_s <- data_input$y_vl
  y_ts_s <- data_input$y_ts
  
  # Escalar las características (features) usando vectorización
  for (i in 1:NFEATS) {
    # Extraer los datos para la característica actual
    x_tr_feat <- as.vector(data_input$x_tr[, , i])
    
    # Calcular la mediana y el IQR
    median_x <- median(x_tr_feat)
    IQR_x <- IQR(x_tr_feat)
    
    # Evitar división por cero
    if (IQR_x == 0) {
      IQR_x <- .Machine$double.eps
    }
    
    # Almacenar los parámetros de escalamiento
    scalers_x[[i]] <- list('median' = median_x, 'IQR' = IQR_x)
    
    # Escalar los conjuntos de datos
    x_tr_s[, , i] <- (data_input$x_tr[, , i] - median_x) / IQR_x
    x_vl_s[, , i] <- (data_input$x_vl[, , i] - median_x) / IQR_x
    x_ts_s[, , i] <- (data_input$x_ts[, , i] - median_x) / IQR_x
  }
  
  # Escalar la variable objetivo y
  y_tr_vector <- as.vector(data_input$y_tr)
  median_y <- median(y_tr_vector)
  IQR_y <- IQR(y_tr_vector)
  
  if (IQR_y == 0) {
    IQR_y <- .Machine$double.eps
  }
  
  # Almacenar los parámetros de escalamiento para y
  scaler_y <- list('median' = median_y, 'IQR' = IQR_y)
  
  # Escalar y
  y_tr_s <- (data_input$y_tr - median_y) / IQR_y
  y_vl_s <- (data_input$y_vl - median_y) / IQR_y
  y_ts_s <- (data_input$y_ts - median_y) / IQR_y
  
  # Combinar los datos escalados en una lista
  data_scaled <- list(
    'x_tr_s' = x_tr_s, 'y_tr_s' = y_tr_s,
    'x_vl_s' = x_vl_s, 'y_vl_s' = y_vl_s,
    'x_ts_s' = x_ts_s, 'y_ts_s' = y_ts_s
  )
  
  # Retornar los datos escalados y los parámetros de escalamiento
  return(list('data_scaled' = data_scaled, 'scalers' = list('scalers_x' = scalers_x, 'scaler_y' = scaler_y)))
}

# desescalar los datos
desescalar <- function(matriz, scaler) {
  matriz * scaler$IQR + scaler$median
}

################## 2. Grafico del entrenamiento y validación ###################

# Grafica personalizada 
plot_history <- function(history) {

  df_history <- data.frame(
    epoch = 1:length(history$metrics$loss),   
    loss = history$metrics$loss,
    val_loss = history$metrics$val_loss
  )
  
  df_history <- tidyr::gather(df_history, key = "metric", value = "value", -epoch)
  
  df_history$data <- ifelse(df_history$metric == "loss", "training", "validation")
  
  p <- ggplot(df_history, aes(x = epoch, y = value, color = data, fill = data, linetype = data, shape = data)) +
    geom_point(shape = 21, col = 1, na.rm = TRUE) + 
    geom_smooth(se = FALSE, method = 'loess', na.rm = TRUE, formula = y ~ x) +
    scale_x_continuous(breaks = pretty(df_history$epoch)[pretty(df_history$epoch) %% 1 == 0]) +
    theme(
      axis.title.y = element_text(size = 12), 
      axis.title.x = element_text(size = 12),  
      strip.placement = 'outside',
      strip.text = element_text(colour = 'black', size = 11),
      strip.background = element_rect(fill = NA, color = NA),
      legend.position = c(0.90, 0.90),  
      legend.box.margin = margin(0, 0, 0, 0)  
    ) +
    scale_color_manual(values = c("training" = "#1f77b4", "validation" = "#ff7f0e")) +
    scale_fill_manual(values = c("training" = "#1f77b4", "validation" = "#ff7f0e")) +
    guides(
      color = guide_legend(title = NULL),
      fill = guide_legend(title = NULL),
      linetype = guide_legend(title = NULL),
      shape = guide_legend(title = NULL)
    ) +
    labs(
      y = "Loss",  # Añadir nombre al eje Y
      x = "Epoch"  # Nombre del eje X
    )
  
  # Imprimir el gráfico
  print(p)
}


####################### 4. Evaluación del modelo sobre Test ####################

# grafica personalizada para ver las secuencias deslizadas del output para el test
plot_output_sequential <- function(y_ts_pred, y_ts_true) {
  ts_length <- length(splits$test)  
  fechas_test <- tail(df$Date, ts_length)  
  num_secuencias <- nrow(y_ts_pred)
  num_pasos_por_secuencia <- ncol(y_ts_pred)  
  fechas_pronosticos <- tail(fechas_test, num_secuencias + num_pasos_por_secuencia - 1)
  primera_fecha <- fechas_pronosticos[1]
  
  # Función auxiliar para convertir matrices a formato largo
  convert_long <- function(df, tipo) {
    df %>%
      mutate(sample = factor(row_number())) %>%               
      pivot_longer(cols = -sample, names_to = "step", values_to = "Valor") %>% 
      mutate(step = as.numeric(gsub("V", "", step)),         
             Tipo = tipo)                                    
  }
  
  # Convertir matrices observadas y predicciones a formato largo
  observado_long <- convert_long(as.data.frame(y_ts_true), "Observado")
  prediccion_long <- convert_long(as.data.frame(y_ts_pred), "Predicción")
  
  # Combinar los data frames observados y predicciones
  data_combined <- bind_rows(observado_long, prediccion_long) %>%
    mutate(
      fecha = as.Date(primera_fecha) + months(as.numeric(as.character(sample)) - 1) + months(step - 1),  # Asignar fechas
      group = ifelse(Tipo == "Observado", "Observado", paste("Secuencia", as.numeric(as.character(sample))))
    )
  
  # Determinar el número de secuencias presentes en los datos
  num_secuencias <- length(unique(data_combined$sample))
  
  # Asignar etiquetas a las muestras secuenciales
  levels(data_combined$sample) <- paste("Secuencia", 1:num_secuencias)
  data_combined$group <- factor(data_combined$group, levels = c("Observado", paste("Secuencia", 1:num_secuencias)))
  
  # Crear una paleta de colores en función del número de secuencias
  palette_pred <- hue_pal()(num_secuencias)
  names(palette_pred) <- paste("Secuencia", 1:num_secuencias)
  
  # Definir paleta de colores combinada
  palette_combined <- c("Observado" = "brown", palette_pred)
  
  # Crear el gráfico
  p <- ggplot(data_combined, aes(x = fecha, y = Valor, group = group, color = group)) +
    geom_line(linewidth = 0.8) +  
    geom_point(data = filter(data_combined, Tipo == "Observado"), size = 2) +
    scale_color_manual(values = palette_combined, name = "Muestra") +
    scale_x_date(date_labels = "%b-%y", date_breaks = "4 months") +
    labs(title = "LSTM Many-to-Many (univariada Multi-Step)",
         x = "Fecha",
         y = "Producción de Mango (t)") +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 1, size = 7),
      plot.title = element_text(size = 12, hjust = 0.5),
      legend.position = "right",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 8)
    )
  
  # Imprimir el gráfico
  print(p)
}

# ver los pronosticos finales con las secuencias promedidas para el pronostio final


# solo observado y predicción promedio
resultLSTM <- function(y_ts_pred, y_ts_true) {
  ts_length <- length(splits$test)  
  fechas_test <- tail(df$Date, ts_length)  
  num_secuencias <- nrow(y_ts_pred)
  num_pasos_por_secuencia <- ncol(y_ts_pred)  
  fechas_pronosticos <- tail(fechas_test, num_secuencias + num_pasos_por_secuencia - 1)
  Inicio <- fechas_pronosticos[1]
  
  convert_long <- function(df, tipo) {
    df %>%
      mutate(sample = factor(row_number())) %>%              
      pivot_longer(cols = -sample, names_to = "step", values_to = "Valor") %>%  
      mutate(step = as.numeric(gsub("V", "", step)),        
             Tipo = tipo)                                    
  }
  
  observado_long <- convert_long(as.data.frame(y_ts_true), "Observado")
  prediccion_long <- convert_long(as.data.frame(y_ts_pred), "Predicción")
  
  data_combined <- bind_rows(observado_long, prediccion_long) %>%
    mutate(
      fecha = as.Date(Inicio) + months(as.numeric(as.character(sample)) - 1) + months(step - 1),  # Asignar fechas
      group = ifelse(Tipo == "Observado", "Observado", paste("Secuencia", as.numeric(as.character(sample))))
    )
  
  observados <- filter(data_combined, Tipo == "Observado")
  predicciones <- filter(data_combined, Tipo == "Predicción")
  
  predicciones_promedio <- predicciones %>%
    group_by(fecha) %>%
    summarise(Predicción = mean(Valor, na.rm = TRUE))  
  
 
  resultados <- full_join(observados, predicciones_promedio, by = "fecha") %>%
    mutate(
      Observado = coalesce(Valor, 0),  
      Predicción = coalesce(Predicción, 0)
    )
  
  return(resultados)
}
plot_resultLSTM <- function(resultados) {
  p <- ggplot(resultados, aes(x = fecha)) +
    geom_line(aes(y = Observado, color = "Test"), linewidth = 1) +  
    geom_point(aes(y = Observado, color = "Test"), size = 2) +  
    geom_line(aes(y = Predicción, color = "Predicción"), linewidth = 1) +  
    geom_point(aes(y = Predicción, color = "Predicción"), size = 2) +
    scale_color_manual(values = c("Test" = "#008B00", "Predicción" = "#DAA520")) +
    labs(title = "Producción Observada (test) y Predicciones",
         x = "Fecha", y = "Producción de Mango (t)") +
    theme(
      legend.position = c(0.95, 0.95),  
      legend.justification = c("right", "top"), 
      legend.title = element_blank(),
      legend.background = element_rect(fill = "white", colour = "white"),  
      legend.key = element_rect(fill = "white", colour = "white"),  
      axis.text.x = element_text(angle = 0, hjust = 1),
      plot.margin = unit(c(1, 1, 1, 1), "lines"),  
      plot.title = element_text(size = 12)  
    )
  
  # Imprimir el gráfico
  print(p)
}


# test completo y observados , pronosticados 
resultLSTM_testC <- function(y_ts_pred, y_ts_true) {
  ts_length <- length(splits$test)  
  fechas_test <- tail(df$Date, ts_length)  
  num_secuencias <- nrow(y_ts_pred)
  num_pasos_por_secuencia <- ncol(y_ts_pred)  
  fechas_pronosticos <- tail(fechas_test, num_secuencias + num_pasos_por_secuencia - 1)
  Inicio <- fechas_pronosticos[1]
  
  # Función auxiliar para convertir matrices a formato largo
  convert_long <- function(df, tipo) {
    df %>%
      mutate(sample = factor(row_number())) %>%               
      pivot_longer(cols = -sample, names_to = "step", values_to = "Valor") %>%  
      mutate(step = as.numeric(gsub("V", "", step)),        
             Tipo = tipo)                                   
  }
  
  observado_long <- convert_long(as.data.frame(y_ts_true), "Observado")
  prediccion_long <- convert_long(as.data.frame(y_ts_pred), "Predicción")

  data_combined <- bind_rows(observado_long, prediccion_long) %>%
    mutate(
      fecha = as.Date(Inicio) + months(as.numeric(as.character(sample)) - 1) + months(step - 1),  # Asignar fechas
      group = ifelse(Tipo == "Observado", "Observado", paste("Secuencia", as.numeric(as.character(sample))))
    )

  observados <- filter(data_combined, Tipo == "Observado")
  predicciones <- filter(data_combined, Tipo == "Predicción")
  
  predicciones_promedio <- predicciones %>%
    group_by(fecha) %>%
    summarise(Predicción = mean(Valor, na.rm = TRUE))  
  
  resultados <- full_join(observados, predicciones_promedio, by = "fecha") %>%
    mutate(
      Observado = coalesce(Valor, 0), 
      Predicción = coalesce(Predicción, 0)
    )
  
  test_completo <- data.frame(fecha = fechas_test, Observado = splits$test, Predicción = NA)
  
  return(list(test_completo = test_completo, resultados = resultados))
}

plot_resultLSTM_testC <- function(test_completo, resultados) {
  
  p <- ggplot() +
    geom_line(data = test_completo, aes(x = fecha, y = Observado, color = "Test (imput)"), linewidth = 1) +
    geom_line(data = resultados, aes(x = fecha, y = Observado, color = "Test"), linewidth = 1) +
    geom_point(data = resultados, aes(x = fecha, y = Observado, color = "Test"), size = 2) +  # Asegurar el mismo color para 'Test'
    geom_line(data = resultados, aes(x = fecha, y = Predicción, color = "Predicción"), linewidth = 1) +
    geom_point(data = resultados, aes(x = fecha, y = Predicción, color = "Predicción"), size = 2) +  # Asegurar el mismo color para 'Predicción'
    scale_color_manual(values = c("Test (imput)" = "#008B8B", "Test" = "#008B00", "Predicción" = "#DAA520")) +
    labs(title = "Producción Observada (test) y Predicciones",
         x = "Fecha", y = "Producción de Mango (t)") +
    scale_x_date(date_labels = "%b %y", date_breaks = "6 months") +  
    theme(
      legend.position = c(0.95, 0.95),  
      legend.justification = c("right", "top"),  
      legend.title = element_blank(),
      legend.background = element_rect(fill = "white", colour = "white"),  
      legend.key = element_rect(fill = "white", colour = "white"),  
      axis.text.x = element_text(angle = 0, hjust = 0.5),  
      plot.margin = unit(c(1, 1, 1, 1), "lines"),  
      plot.title = element_text(size = 10)  
    )
  
  print(p)
}


resultLSTM_complete <- function(y_ts_pred, y_ts_true) {
  ts_length <- length(splits$test)  
  fechas_test <- tail(df$Date, ts_length)  
  num_secuencias <- nrow(y_ts_pred)
  num_pasos_por_secuencia <- ncol(y_ts_pred)  
  fechas_pronosticos <- tail(fechas_test, num_secuencias + num_pasos_por_secuencia - 1)
  Inicio <- fechas_pronosticos[1]
  

  convert_long <- function(df, tipo) {
    df %>%
      mutate(sample = factor(row_number())) %>%               
      pivot_longer(cols = -sample, names_to = "step", values_to = "Valor") %>%  
      mutate(step = as.numeric(gsub("V", "", step)),         
             Tipo = tipo)                                    
  }
  
  observado_long <- convert_long(as.data.frame(y_ts_true), "Observado")
  prediccion_long <- convert_long(as.data.frame(y_ts_pred), "Predicción")
  

  data_combined <- bind_rows(observado_long, prediccion_long) %>%
    mutate(
      fecha = as.Date(Inicio) + months(as.numeric(as.character(sample)) - 1) + months(step - 1), 
      group = ifelse(Tipo == "Observado", "Observado", paste("Secuencia", as.numeric(as.character(sample))))
    )
  

  observados <- filter(data_combined, Tipo == "Observado")
  predicciones <- filter(data_combined, Tipo == "Predicción")
  
  
  predicciones_promedio <- predicciones %>%
    group_by(fecha) %>%
    summarise(Predicción = mean(Valor, na.rm = TRUE))  
  

  resultados <- full_join(observados, predicciones_promedio, by = "fecha") %>%
    mutate(
      Observado = coalesce(Valor, 0),  
      Predicción = coalesce(Predicción, 0)
    )
  

  test_completo <- data.frame(fecha = fechas_test, Observado = splits$test, Predicción = NA)
  

  full_data <- data.frame(
    fecha = df$Date,
    Observado = df$Producción
  )
  
  return(list(full_data = full_data, test_completo = test_completo, resultados = resultados))
}
plot_resultLSTM_complete <- function(full_data, test_completo, resultados) {
  
  # Graficar la serie completa, el conjunto de prueba y los pronósticos promediados
  p <- ggplot() +
    geom_line(data = full_data, aes(x = fecha, y = Observado, color = "Producción(t)"), linewidth = 1) +  # Serie completa
    geom_line(data = test_completo, aes(x = fecha, y = Observado, color = "Test (input)"), linewidth = 1) +  # Test
    geom_line(data = resultados, aes(x = fecha, y = Observado, color = "Test"), linewidth = 1) +  # Observado del test
    geom_line(data = resultados, aes(x = fecha, y = Predicción, color = "Predicción"), linewidth = 1) +  # Predicción promediada
    scale_color_manual(values = c("Producción(t)" = "#AE123A", "Test (input)" = "#008B8B", "Test" = "#008B00", "Predicción" = "#DAA520")) +
    labs(title = "Producción(t), Producción Observada (test) y Predicciones",
         x = "Fecha", y = "Producción de Mango (t)") +
    scale_x_date(date_labels = "%b %y", date_breaks = "32 months") +  # Mostrar cada 6 meses
    theme(
      legend.position = c(0.28, 0.97),  # Ajustar la posición dentro del gráfico con coordenadas normalizadas
      legend.justification = c("right", "top"),  # Alineación de la leyenda
      legend.title = element_blank(),
      legend.background = element_rect(fill = "white", colour = "white"),  # Fondo blanco
      legend.key = element_rect(fill = "white", colour = "white"),  # Claves con fondo blanco
      axis.text.x = element_text(angle = 0, hjust = 0.5),  # Alinear horizontalmente las etiquetas
      plot.margin = unit(c(1, 1, 1, 1), "lines"),  # Ajustar márgenes si es necesario
      plot.title = element_text(size = 10)  # Ajustar el tamaño del título
    )
  
  # Imprimir el gráfico
  print(p)
}


tibble_pre_obs <- function(y_ts_true, y_ts_pred, df) {
  ts_length <- length(df$Date) - nrow(y_ts_pred)
  fechas_test <- tail(df$Date, ts_length)
  
  num_secuencias <- nrow(y_ts_pred)
  num_pasos_por_secuencia <- ncol(y_ts_pred)  
  fechas_pronosticos <- tail(fechas_test, num_secuencias + num_pasos_por_secuencia - 1)
  Inicio <- fechas_pronosticos[1]
  
  #
  convert_long <- function(df, tipo) {
    df %>%
      mutate(sample = factor(row_number())) %>%               
      pivot_longer(cols = -sample, names_to = "step", values_to = "Valor") %>%  
      mutate(step = as.numeric(gsub("V", "", step)),         
             Tipo = tipo)                                    
  }
  
  observado_long <- convert_long(as.data.frame(y_ts_true), "Observado")
  prediccion_long <- convert_long(as.data.frame(y_ts_pred), "Predicción")
  
  data_combined <- bind_rows(observado_long, prediccion_long) %>%
    mutate(
      fecha = as.Date(Inicio) + months(as.numeric(as.character(sample)) - 1) + months(step - 1)  
    ) %>%
    group_by(fecha, Tipo) %>%
    summarise(Valor = mean(Valor, na.rm = TRUE), .groups = 'drop')  %>% 
    pivot_wider(names_from = Tipo, values_from = Valor)
  
  return(data_combined)
}
metrics <- function(observed, predicted) {
  # Calcular métricas
  mase_value <- Metrics::mase(actual = observed, predicted = predicted, step_size = 12)
  mae_value <- Metrics::mae(observed, predicted)
  mdae_value <- Metrics::mdae(observed, predicted)
  smape_value <- Metrics::smape(observed, predicted)
  
  # Formatear los valores con 3 decimales usando sprintf para consistencia
  mase_formatted <- sprintf("%.3f", mase_value)
  mae_formatted <- sprintf("%.3f", mae_value)
  mdae_formatted <- sprintf("%.3f", mdae_value)
  smape_formatted <- sprintf("%.3f", smape_value)
  
  # Crear una tabla de métricas
  metrics_table <- data.frame(
    Metric = c("MASE", "MAE", "MDAE", "sMAPE"),
    Value = c(mase_formatted, mae_formatted, mdae_formatted, smape_formatted)
  )
  
  # Imprimir la tabla de métricas
  print(metrics_table)
}

#===============================================================================
# Bibliografia CODIGO R LSTM
# https://machinelearningmastery.com/multi-step-time-series-forecasting/
# https://youtu.be/TEzTfl_E-3o?si=fmB77hrkS42O_-fr
# https://medium.com/analytics-vidhya/undestanding-recurrent-neural-network-rnn-and-long-short-term-memory-lstm-30bc1221e80d
# https://www.tensorflow.org/tutorials/structured_data/time_series?hl=es-419
# https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/ (alternativa recursiva)
# https://www.deeplearningbook.org/

# Si hubiera sido el caso de que ganaba el modelo LSTM al SARIMA se predecia con:

##################### 5. Predicción fuera de la Muestra ########################

# Ojo solo predice el OUTPUT_LENGTH que se configura desde el inicio 

# 5.1. Cargar el modelo entrenado y guardado antes usando keras3
predic_modelo <- keras3::load_model("LSTM_A.keras")


# 5.2. Preparar los últimos datos para la predicción
ultimos_datos <- tail(df$Producción, INPUT_LENGTH)  # Obtener los últimos 24 valores
ultimos_datos_s <- (ultimos_datos - scalers$scaler_y$median) / scalers$scaler_y$IQR
ultimos_datos_s <- array(ultimos_datos_s, dim = c(1, INPUT_LENGTH, 1))

# 5.3. Hacer el pronóstico
prediccion_futuro_s <- predic_modelo %>% predict(ultimos_datos_s)
prediccion_futuro <- desescalar(prediccion_futuro_s, scalers$scaler_y)

print(prediccion_futuro)

# 5.4. Visualizo pronósticos

# Crear un data frame con los resultados de la predicción
fechas_futuras <- seq(as.Date("2024-09-01"), by = "month", length.out = length(prediccion_futuro))  # Fechas futuras para el pronóstico
df_prediccion <- data.frame(
  Fecha = fechas_futuras,
  Prediccion = as.vector(prediccion_futuro)
)

# Graficar con ggplot2
library(ggplot2)

ggplot(df_prediccion, aes(x = Fecha, y = Prediccion)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +  
  labs(title = "Predicciones de Producción Futura",
       x = "Fecha",
       y = "Producción de Mango (t)") +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


library(ggplot2)

p <- ggplot() +
  geom_line(data = df, aes(x = Date, y = Producción, color = "Producción"), size = 1) + 
  geom_line(data = df_prediccion, aes(x = Fecha, y = Prediccion, color = "Pronóstico"), size = 1) +  
  labs(title = "Producción Histórica y Pronósticos de Mango",
       x = "Fecha",
       y = "Producción de Mango (t)") +
  scale_color_manual(values = c("Producción" = "#AE123A", "Pronóstico" = "#53868B")) + 
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_blank(),  
    legend.position = c(0.2, 0.8),  
    legend.background = element_rect(fill = "white", color = "black"),  
    legend.box.background = element_rect(fill = "white", colour = "black"),
    legend.key = element_rect(fill = "white") 
  )
print(p)















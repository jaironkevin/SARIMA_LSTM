
library(urca)         # Para los tests ADF y KPSS.
library(knitr)        # Para generar tablas (función kable).
library(kableExtra)   # Para estilizar tablas generadas con kable.
library(forecast)     # Para análisis y predicción de series temporales.
library(tseries)      # Para análisis de series temporales (e.g., test de Ljung-Box).
library(ggplot2)      # Para gráficos (ACF, PACF, etc.).
library(gridExtra)    # Para organizar múltiples gráficos en un mismo layout.


# Funciónes para descriptivos del OE1

# Moda
Moda <- function(x) {  
  q <- unique(x)
  q[which.max(tabulate(match(x, q)))]
}

# coeficiente de variación
CV <- function(x) { 
  y <- 100 * sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE)
  return(y)
}

# Funciones para probar la estacionariedad o no de la serie
adf_table <- function(time_series) {
  
  adf_test <- ur.df(time_series, selectlags = "AIC", type = "none")
  summary_adf <- summary(adf_test)
  
  test_statistic <- summary_adf@teststat
  critical_values <- summary_adf@cval
  
  decision <- ifelse(test_statistic < critical_values["tau1", "5pct"], "Estacionaria", "No Estacionaria")
  
  results_table <- data.frame(
    Description = c("Augmented Dickey-Fuller (ADF)", 
                    "Critical Value for: 1%", "5%", 
                    "10%"),
    Value = c(
      test_statistic,  
      critical_values["tau1", "1pct"],  
      critical_values["tau1", "5pct"],  
      critical_values["tau1", "10pct"]  
    )
  )
  
  kable(results_table, format = "html", 
        col.names = c("Test", "t-Statistic"), align = c('r', 'c')) %>%
    kable_styling(full_width = FALSE, position = "center") %>%
    column_spec(1, bold = TRUE) %>%
    row_spec(1, extra_css = "border-bottom: 1px solid black;") %>%
    row_spec(2, extra_css = "text-align: right;") %>%
    row_spec(3, extra_css = "text-align: right;") %>%
    row_spec(4, extra_css = "border-bottom: 1px solid black; text-align: right;")
}

kpss_table <- function(time_series) {
  # Asegurarse de que la serie temporal no tenga valores faltantes
  if (any(is.na(time_series))) {
    stop("La serie temporal contiene valores faltantes. Por favor, maneje los valores faltantes antes de realizar el test.")
  }
  
  if (!is.numeric(time_series)) {
    stop("La serie temporal debe ser numérica.")
  }
  
  # Realizar el test KPSS
  kpss_test <- ur.kpss(time_series, type = "tau", use.lag = NULL)
  summary_kpss <- summary(kpss_test)
  
  test_statistic <- summary_kpss@teststat
  critical_values10 <- summary_kpss@cval[1]
  critical_values05 <- summary_kpss@cval[2]
  critical_values2.5 <- summary_kpss@cval[3]
  critical_values1 <- summary_kpss@cval[4]
  
  decision <- ifelse(test_statistic < critical_values1, "Estacionaria", "No Estacionaria")
  
  results_table <- data.frame(
    Description = c("Kwiatkowski–Phillips–Schmidt–Shin (KPSS)", 
                    "Critical Value for: 10%", "5%", 
                    "2.5%", "1%"),
    Value = c(
      test_statistic,  
      critical_values10,  
      critical_values05,  
      critical_values2.5,  
      critical_values1  
    )
  )
  
  kable(results_table, format = "html", 
        col.names = c("Test", "Statistic"), align = c('r', 'c')) %>%
    kable_styling(full_width = FALSE, position = "center") %>%
    column_spec(1, bold = TRUE) %>%
    row_spec(1, extra_css = "border-bottom: 1px solid black;") %>%
    row_spec(2, extra_css = "text-align: right;") %>%
    row_spec(3, extra_css = "text-align: right;") %>%
    row_spec(4, extra_css = "text-align: right;") %>%
    row_spec(5, extra_css = "border-bottom: 1px solid black; text-align: right;")
}

# Funciones para la identificación de ordenes del modelo SARIMA
# AFC para identificación
afc.id <- function(data, lag.max, period, threshold) {
  # Calcula la ACF
  acf_result <- acf(data, lag.max = lag.max, plot = FALSE)
  colors_acf <- rep("#2686A0", lag.max + 1)  
  
  # Resalta los lags que superan el umbral y corresponden al período
  for (i in 2:lag.max) {  # Comienza desde el segundo lag
    if (abs(acf_result$acf[i]) > 1.95/sqrt(length(data)) && (i - 1) %% period == 0) {
      colors_acf[i] <- "#D64267"
    } else {
      colors_acf[i] <- "#2686A0"  
    }
  }
  
  library(ggplot2)
  
  p <- ggplot() +
    xlim(0, lag.max) +
    labs(x = "Lag", y = "ACF") +
    geom_rect(aes(xmin = 0, xmax = lag.max, ymin = -1.96/sqrt(length(data)), ymax = 1.96/sqrt(length(data))), 
              fill = rgb(139/255, 69/255, 19/255, alpha = 0.2), color = NA) +
    geom_hline(yintercept = c(-1.96/sqrt(length(data)), 0, 1.96/sqrt(length(data))), 
               linetype = "dashed", color = "brown") +
    geom_hline(yintercept = 0, linetype = "solid", color = "black") +
    geom_segment(aes(x = 1:lag.max, y = 0, xend = 1:lag.max, yend = acf_result$acf[2:(lag.max + 1)]), 
                 color = colors_acf[-1], size = 1, linetype = "solid") +
    geom_point(aes(x = 1:lag.max, y = acf_result$acf[2:(lag.max + 1)]), 
               shape = 21, size = 3, fill = colors_acf[-1])
  
  print(p)
}

# PACF para identificación
pacf.id <- function(data, lag.max, period, threshold) {
  # Calcula la PACF
  pacf_result <- pacf(data, lag.max = lag.max, plot = FALSE)
  
  # Resalta los lags que superan el umbral y corresponden al período
  colors_pacf <- rep("#2686A0", lag.max)
  
  for (i in 1:lag.max) {  
    if (abs(pacf_result$acf[i]) > 1.95/sqrt(length(data)) && i %% period == 0) {
      colors_pacf[i] <- "#D64267"
    }
  }
  
  library(ggplot2)
  
  # Gráfico de PACF
  p_pacf <- ggplot() +
    xlim(0, lag.max) +
    labs(x = "Lag", y = "PACF") +
    geom_rect(aes(xmin = 0, xmax = lag.max, ymin = -1.96/sqrt(length(data)), ymax = 1.96/sqrt(length(data))), 
              fill = rgb(139/255, 69/255, 19/255, alpha = 0.2), color = NA) +
    geom_hline(yintercept = c(-1.96/sqrt(length(data)), 0, 1.96/sqrt(length(data))), 
               linetype = "dashed", color = "brown") +
    geom_hline(yintercept = 0, linetype = "solid", color = "black") +
    geom_segment(aes(x = 1:lag.max, y = 0, xend = 1:lag.max, yend = pacf_result$acf[1:lag.max]), 
                 color = colors_pacf, size = 1, linetype = "solid") +
    geom_point(aes(x = 1:lag.max, y = pacf_result$acf[1:lag.max]), 
               shape = 21, size = 3, fill = colors_pacf)
  
  print(p_pacf)
}



# Funciónes para diagnostico de residuos ARIMA estacional
LjBox.Pierce_Inde <- function(model, max.lag = 25, type = c("Ljung-Box", "Box-Pierce")) {
  type <- match.arg(type)
  
  # Aqui extraigo los residuos
  residuos <- resid(model)
  
  #  ACF y PACF
  acf_resid <- acf(residuos, type = "correlation", plot = FALSE, lag.max = max.lag)
  pacf_resid <- pacf(residuos, plot = FALSE, lag.max = max.lag)
  
  # Genero las tabla de ACF y PACF
  Lag <- 1:max.lag
  acf_table <- data.frame(ACF = round(acf_resid$acf[-1], 3))  
  pacf_table <- data.frame(PACF = round(pacf_resid$acf, 3))  
  
  # Prueba de Ljung-Box o Box-Pierce
  p.value <- rep(NA, max.lag)
  Qm <- rep(NA, max.lag)
  fit <- sum(arimaorder(model)[c("p", "q", "P", "Q")], na.rm = TRUE)
  
  for (i in 1:max.lag) {
    if (i > fit) {
      test <- Box.test(residuos, type = type, fitdf = fit, lag = i)
      p.value[i] <- test$p.value
      Qm[i] <- test$statistic
    } else {
      test <- Box.test(residuos, type = type, fitdf = 0, lag = i)
      Qm[i] <- test$statistic
    }
  }
  
  # Redondeo a 3 decimales
  Qm <- round(Qm, 3)
  p.value <- round(p.value, 3)
  
  autocorres <- data.frame(Lag, ACF = round(acf_table$ACF, 3), PACF = pacf_table$PACF, Qm, pvalue = p.value)
  autocorres_print <- autocorres
  autocorres_print[is.na(autocorres_print)] <- " "
  
  return(autocorres)
}

check_residuals <- function(model, lag.max = 25, title = "Modelo SARIMA", type = "Ljung-Box") {
  diff_regular <- model$arma[6]  # d
  diffs_estacional <- model$arma[5] * model$arma[7]  # D * Seasonal period
  n_excluded <- diff_regular + diffs_estacional
  
  # Obtener residuos y su frecuencia
  residuos <- residuals(model)
  frequency_of_data <- frequency(residuos)
  
  # Excluir los primeros n_excluded residuos, si hay suficientes
  if (length(residuos) > n_excluded) {
    start_time <- time(residuos)[n_excluded + 1]
    residuos <- window(residuos, start = start_time)
  } else {
    residuos <- residuos
  }
  
  # Gráfico de los residuos
  P5 <- autoplot(residuos, series = "Residuos", color = "#8B8B00") +
    geom_point(aes(y = residuos), size = 1, color = "#09622A") + 
    labs(y = "Residuos", x = "Índice") +
    ggtitle(paste("Residuos del", title)) +
    theme(
      plot.title = element_text(hjust = 0.5),
      axis.text.x = element_text(angle = 0, hjust = 1)
    )
  
  # Función para graficar ACF de residuos
  afc.re <- function(data, lag.max) {
    acf_result <- acf(data, lag.max = lag.max, plot = FALSE)
    significant <- abs(acf_result$acf) > 1.96/sqrt(length(data))
    
    p <- ggplot() +
      geom_blank() +
      xlim(0, lag.max) +
      labs(x = "Lag", y = "ACF") +
      geom_rect(aes(xmin = 0, xmax = lag.max, ymin = -1.96/sqrt(length(data)), ymax = 1.96/sqrt(length(data))), 
                fill = rgb(139/255, 69/255, 19/255, alpha = 0.2), color = NA) +
      geom_hline(yintercept = c(-1.96/sqrt(length(data)), 0, 1.96/sqrt(length(data))), 
                 linetype = "dashed", color = "brown") +
      geom_hline(yintercept = 0, linetype = "solid", color = "black") +
      geom_segment(aes(x = 1:lag.max, y = 0, xend = 1:lag.max, yend = acf_result$acf[2:(lag.max + 1)]), 
                   color = ifelse(significant[2:(lag.max + 1)], "black", "#005F52"), size = 1, linetype = "solid") +
      geom_point(aes(x = 1:lag.max, y = acf_result$acf[2:(lag.max + 1)]), 
                 shape = 21, size = 3, fill = ifelse(significant[2:(lag.max + 1)], "black", "#005F52")) +
      expand_limits(y = c(-1, 1)) +  
      coord_cartesian(ylim = c(-0.25, 0.25))
    
    return(p)
  }
  
  P6 <- afc.re(residuals(model), lag.max = lag.max)
  
  # Histograma de los residuos
  histores <- as.vector(residuos)
  residh <- as.data.frame(histores)
  
  P8 <- ggplot(residh, aes(x = histores)) +
    geom_histogram(bins = 60, aes(y = after_stat(density)),
                   colour = "black", fill = "#BD5DA6") +
    labs(y = 'Frecuencia', x = "Residuos del modelo") +
    geom_rug() +
    stat_function(fun = dnorm, args = list(mean = mean(residh$histores), sd = sd(residh$histores)), 
                  geom = "area", fill = "black", alpha = 0.2, linetype = "solid", linewidth = 0.5, colour = "darkred")
  
  # Función Ljung-Box
  LjBox.Pierce_Inde <- function(model, max.lag = 25, type = type) {
    type <- match.arg(type)
    residuos <- resid(model)
    p.value <- rep(NA, max.lag)
    Qm <- rep(NA, max.lag)
    fit <- sum(arimaorder(model)[c("p", "q", "P", "Q")], na.rm = TRUE)
    
    for (i in 1:max.lag) {
      if (i > fit) {
        test <- Box.test(residuos, type = type, fitdf = fit, lag = i)
        p.value[i] <- test$p.value
        Qm[i] <- test$statistic
      } else {
        test <- Box.test(residuos, type = type, fitdf = 0, lag = i)
        Qm[i] <- test$statistic
      }
    }
    
    Qm <- round(Qm, 3)
    p.value <- round(p.value, 3)
    autocorres <- data.frame(Lag = 1:max.lag, Qm, pvalue = p.value)
    
    return(autocorres)
  }
  
  LJ <- LjBox.Pierce_Inde(model, max.lag = lag.max, type = type)
  LJ_valid <- LJ[!is.na(LJ$pvalue), ]
  
  P9 <- ggplot(LJ_valid, aes(x = Lag, y = pvalue)) +
    geom_point(stat = "identity", size = 2, shape = 16, color = "#4CAF50", fill = "#4CAF50") +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", size = 1) +
    geom_hline(yintercept = 0.01, linetype = "dashed", color = "blue", size = 0.5) +  
    labs(title = "P_values for Ljung-Box statistic", x = "Lag (H)", y = "p-value") +
    scale_x_continuous(breaks = 1:lag.max) +
    theme(plot.title = element_text(hjust = 0.5, size = 10))
  
  
  layout <- rbind(
    c(5, 5),
    c(6, 8),
    c(9, 9)
  )
  
  grid.arrange(
    P5, P6, P8, P9,
    layout_matrix = layout
  )
}

test.residuals <- function(model, test_normal = "Shapiro-Wilk", test_hetero = "Levene", test_ac = "Ljung-Box", test_stationarity = "ADF", test_media = FALSE, nrepl = 2000) {
  # Configuración inicial de los datos
  diff_regular <- model$arma[6]  # d
  diffs_estacional <- model$arma[5] * model$arma[7]  # D * Seasonal period
  n_excluded <- diff_regular + diffs_estacional
  
  # Obtener residuos y su frecuencia
  residuos <- stats::residuals(model)
  frequency_of_data <- stats::frequency(residuos)
  
  # Excluir los primeros n_excluded residuos, si hay suficientes
  if (length(residuos) > n_excluded) {
    start_time <- stats::time(residuos)[n_excluded + 1]
    residuos <- stats::window(residuos, start = start_time)
  }
  
  # Seleccionar la prueba de normalidad
  if (test_normal == "Shapiro-Wilk") {
    test_normalidad <- stats::shapiro.test(residuos)
    test_name <- "Shapiro-Wilk"
  } else if (test_normal == "Jarque-Bera") {
    test_normalidad <- tseries::jarque.bera.test(residuos)
    test_name <- "Jarque-Bera"
  } else if (test_normal == "Kolmogorov-Smirnov") {
    test_normalidad <- stats::ks.test(residuos, "pnorm", mean = mean(residuos), sd = sd(residuos))
    test_name <- "Kolmogorov-Smirnov"
  } else {
    stop("Test normal no reconocido. Elija entre 'Shapiro-Wilk', 'Jarque-Bera', 'Kolmogorov-Smirnov'.")
  }
  
  # Test de independencia de los residuos
  if (test_ac == "Ljung-Box") {
    test_ac_result <- stats::Box.test(resid(model), lag = 24, type = "Ljung-Box")
  } else if (test_ac == "Box-Pierce") {
    test_ac_result <- stats::Box.test(resid(model), lag = 24, type = "Box-Pierce")
  } else {
    stop("Test de autocorrelación no reconocido. Elija entre 'Ljung-Box', 'Box-Pierce'.")
  }
  
  # Test de heterocedasticidad
  if (test_hetero == "Levene") {
    start_year <- stats::start(residuos)[1]
    start_month <- stats::start(residuos)[2]
    freq <- stats::frequency(residuos)
    start_date <- base::as.Date(paste(start_year, start_month, "1", sep = "-"))
    n_months <- length(residuos) / freq * 12
    end_date <- base::as.Date(paste(start_year + floor((start_month + n_months - 1) / 12), ((start_month + n_months - 1) %% 12) + 1, "1", sep = "-"))
    fechas <- base::seq(from = start_date, to = end_date, by = "month")
    
    if (length(fechas) > length(residuos)) {
      fechas <- fechas[1:length(residuos)]
    }
    
    residuos_zoo <- zoo::zoo(residuos, order.by = fechas)
    residuos_df <- data.frame(
      Fecha = zoo::index(residuos_zoo),
      Residuos = zoo::coredata(residuos_zoo),
      Grupo = as.factor(base::format(zoo::index(residuos_zoo), "%Y"))
    )
    test_hetero_result <- car::leveneTest(Residuos ~ Grupo, data = residuos_df)
    test_hetero_name <- "Levene"
    hetero_stat <- test_hetero_result$`F value`[1]
    hetero_p_value <- test_hetero_result$`Pr(>F)`[1]
  }
  
  # Prueba de estacionariedad
  if (test_stationarity == "ADF") {
    test_stationarity_result <- suppressWarnings(tseries::adf.test(residuos, alternative = "stationary"))
    test_stationarity_name <- "ADF"
  } else if (test_stationarity == "KPSS") {
    test_stationarity_result <- suppressWarnings(tseries::kpss.test(residuos))
    test_stationarity_name <- "KPSS"
  } else {
    stop("Test de estacionariedad no reconocido. Elija entre 'ADF', 'KPSS'.")
  }
  
  # Realizar la prueba de media si se solicita
  if (test_media) {
    t_test_result <- stats::t.test(residuos, mu = 0)
    media_t_value <- unname(t_test_result$statistic)
    media_p_value <- t_test_result$p.value
    media_decision <- ifelse(media_p_value < 0.05, "Difiere de 0", "No difiere de 0")
    media_test_name <- "t de Media"
  }
  
  # Compilado de resultados
  resultados <- data.frame(
    Evaluacion = c("Independencia", "Normalidad", "Homocedasticidad", "Estacionariedad"),
    Test = c(test_ac, test_name, test_hetero_name, test_stationarity_name),
    Estadistico = sprintf("%.3f", c(as.numeric(test_ac_result$statistic), as.numeric(test_normalidad$statistic), as.numeric(hetero_stat), as.numeric(test_stationarity_result$statistic))),
    P_valor = sprintf("%.3f", c(test_ac_result$p.value, test_normalidad$p.value, hetero_p_value, test_stationarity_result$p.value)),
    Decisión = c(
      ifelse(test_ac_result$p.value < 0.05, "No independientes", "Independientes"),
      ifelse(test_normalidad$p.value < 0.05, "No Normal", "Normal"),
      ifelse(hetero_p_value < 0.05, "Heterocedasticidad", "Homocedasticidad"),
      ifelse(test_stationarity == "ADF", ifelse(test_stationarity_result$p.value < 0.05, "Stationary", "Not Stationary"),
             ifelse(test_stationarity_result$p.value < 0.05, "Not Stationary", "Stationary"))
    )
  )
  # Añadir prueba de media si se solicitó y reordenar
  if (test_media) {
    resultados_media <- data.frame(
      Evaluacion = "Media 0",
      Test = media_test_name,
      Estadistico = sprintf("%.3f", media_t_value),
      P_valor = sprintf("%.3f", media_p_value),
      Decisión = media_decision
    )
    resultados <- rbind(resultados_media, resultados)
    # Reordenar para colocar Media antes de Estacionariedad
    orden_deseado <- c(2, 3, 4, 1, 5) # Ajusta según el orden actual y deseado
    resultados <- resultados[orden_deseado, ]
  }
  
  return(resultados)
}









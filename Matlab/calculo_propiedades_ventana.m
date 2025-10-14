function resultados = calculo_propiedades_ventana(x, tam_ventana)
    % Esta función calcula varias propiedades de una señal
    % usando un enfoque de ventana deslizante, sin usar funciones
    % predefinidas de MATLAB para las operaciones.
    % ENTRADAS:
    %   x:             Vector de la señal de entrada.
    %   tam_ventana:   Tamaño de la ventana para el análisis.    
    % SALIDA:
    %   resultados:    Una estructura con los siguientes campos:
    %                  .media, .energia, .magnitud_pico, .varianza, .desviacion

    % Inicializar las variables de resultados
    resultados.media = [];
    resultados.energia = [];
    resultados.magnitud_pico = [];
    resultados.varianza = [];
    resultados.desviacion = [];

    n = size(x, 1);
    m = 1; % Índice de inicio de la primera ventana
    paso = floor(tam_ventana / 2); % Paso para la ventana deslizante

    while (m + tam_ventana - 1 <= n)
        % Extraer la porción de la señal que corresponde a la ventana actual
        y = x(m : m + tam_ventana - 1);
        
        % --- CÁLCULOS MANUALES ---
        
        % 1. Media
        suma_media = 0;
        for i = 1 : tam_ventana
            suma_media = suma_media + y(i);
        end
        media = suma_media / tam_ventana;

        % 2. Energía
        suma_energia = 0;
        for i = 1 : tam_ventana
            suma_energia = suma_energia + y(i)^2;
        end
        energia = suma_energia;

        % 3. Magnitud Pico
        magnitud_pico = 0;
        for i = 1 : tam_ventana
            if abs(y(i)) > magnitud_pico
                magnitud_pico = abs(y(i));
            end
        end

        % 4. Varianza
        suma_varianza = 0;
        for i = 1 : tam_ventana
            suma_varianza = suma_varianza + (y(i) - media)^2;
        end
        varianza = suma_varianza / tam_ventana;

        % 5. Desviación Estándar
        desviacion = sqrt(varianza);

        % --- ALMACENAR LOS RESULTADOS ---
        resultados.media(end+1) = media;
        resultados.energia(end+1) = energia;
        resultados.magnitud_pico(end+1) = magnitud_pico;
        resultados.varianza(end+1) = varianza;
        resultados.desviacion(end+1) = desviacion;

        % Mover al siguiente segmento de la señal
        m = m + paso;
    end
end
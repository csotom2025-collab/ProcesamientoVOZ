function [zadelante,zadelante1,resultados_analisis]=ejemplo_calculo_autocorrelacion()
    cd D:/ProcesaminetoVoz/AudiosVoz/;
    [x,yadelante1]=audioread('jos.wav');
    cd ..
    plot(x)
    tam_ven=5;
    zadelante=xcorr(yadelante1);
    zadelante1=calculo_autocorrelacion(yadelante1,tam_ven);
    resultados_analisis = calculo_propiedades_ventana(x, tam_ven);

    E = sum(abs(x).^2);
    magnitud = abs(x);
    media = mean(x);
    varianza = var(x, 1);
    desviacion_estandar = std(x, 1);
    % Display additional analysis results
    fprintf('energia: %.2f \n', E);
    fprintf('magnitud pico: %.2f \n', max(magnitud));
    fprintf('media: %.2f \n', media);
    fprintf('varianza: %.2f \n', varianza);
    fprintf('desviacion_estandar: %.2f \n', desviacion_estandar);
end

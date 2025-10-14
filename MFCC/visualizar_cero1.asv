function visualizar_cero1()
% VISUALIZAR_CERO1  Carga y visualiza el archivo 'cero1.dat'
%   El script intenta primero cargar el archivo como texto (load). Si falla,
%   intenta leerlo como binario probando varios formatos (float32, int16, etc.).
%
%   Visualizaciones:
%     - Forma de onda (tiempo)
%     - Espectrograma (dB)
%     - Histograma de amplitudes
%
%   Edita la variable `fs` debajo si conoces la frecuencia de muestreo.

% -------------- Parámetros editables -----------------
fs = 16000; % frecuencia de muestreo en Hz - cambia este valor si hace falta
filename = 'cero1.dat';
% ----------------------------------------------------

% Intentar cargar como ASCII/archivo MAT
data = [];
try
    tmp = load(filename);
    % load puede devolver struct si el archivo contiene variables
    if isstruct(tmp)
        f = fieldnames(tmp);
        data = tmp.(f{1});
    else
        data = tmp;
    end
catch
    % no pudo cargar como texto/MAT - intentaremos abrir como binario
    fid = fopen(filename,'rb');
    if fid == -1
        error('No se pudo abrir el archivo: %s', filename);
    end
    % Probar varios formatos binarios comunes
    formats = {'float32','int16','int8','uint8','double'};
    for k = 1:length(formats)
        fseek(fid,0,'bof');
        raw = fread(fid,inf,formats{k});
        if ~isempty(raw)
            data = double(raw);
            format_used = formats{k};
            break;
        end
    end
    fclose(fid);
    if isempty(data)
        error('No se pudieron leer datos del archivo %s (binario)', filename);
    end
end

% Si data tiene más de una columna, tomar la primera (suponemos que puede ser estéreo)
if size(data,2) > 1
    data = data(:,1);
end

% Asegurar vector columna
data = data(:);
N = length(data);
if N == 0
    error('El archivo %s no contiene muestras detectables.', filename);
end

% Mostrar información básica
dur = N / fs;
fprintf('Archivo: %s\nMuestras: %d\nfs = %d Hz\nDuración = %.3f s\n', filename, N, fs, dur);

% Normalizar para visualización si valores demasiado grandes (opcional)
maxabs = max(abs(data));
if maxabs > 0
    data_norm = data / maxabs;
else
    data_norm = data;
end

% Crear figura con 3 subplots

plot((0:N-1)/fs, data_norm, 'k');
xlabel('Tiempo (s)');
ylabel('Amplitud (norm)');
title(sprintf('Forma de onda (%s) - fs=%d Hz, dur=%.3f s', filename, fs, dur));
grid on;

% Espectrograma
figure;
win = round(0.025 * fs);    % ventana 25 ms
hop = round(0.010 * fs);    % avance 10 ms (solapamiento = win-hop)
nfft = 2^nextpow2(win);
[S,F,T,P] = spectrogram(data, win, win-hop, nfft, fs);
PdB = 10*log10(abs(P) + eps);
imagesc(T, F, PdB);
axis xy;
colormap jet;
colorbar;
ylim([0 min(fs/2, 8000)]); % mostrar hasta 8 kHz o fs/2 (ajusta si quieres otra cosa)
ylabel('Frecuencia (Hz)');
xlabel('Tiempo (s)');
title('Espectrograma (dB)');

% Histograma
figure;
histogram(data_norm, 256);
xlabel('Amplitud (norm)');
ylabel('Cuenta');
title('Histograma de amplitudes');

% Mensaje final
fprintf('Visualización generada. Ajusta variable ''fs'' en el script si la señal se ve comprimida/estirada.\n');
end

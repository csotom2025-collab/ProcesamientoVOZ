function process_audio(filename)
    [y, fs] = audioread(filename);
    energia = sum(y.^2);
    cruces = sum(abs(diff(sign(y))) > 0);
    result_file = strcat(filename, '_result.txt');
    fid = fopen(result_file, 'w');
    fprintf(fid, 'Archivo: %s\n', filename);
    fprintf(fid, 'Energy: %.4f\n', energia);
    fprintf(fid, 'Cruces por cero: %d\n', cruces);
    fclose(fid);
end

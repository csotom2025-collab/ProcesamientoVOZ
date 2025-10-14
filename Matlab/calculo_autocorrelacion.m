function z=calculo_autocorrelacion(x,tam_ventana)
    z=[];
    k=0:tam_ventana-1;
    w=0.54-0.46*cos((2*pi/tam_ventana-1)*k);
    a=0.95;
    n=size (x,1);
    y(1)=x(1);
    for k=2:n
        y(k)=x(k)-a*x(k-1);
    end
    x=y';
    m=1;
    while (m+(tam_ventana-1) <= n)
        y=x(m:m+(tam_ventana-1));
        y=w'.*y;
        r=autox(y);
        z=[z,r];
        m=m+((tam_ventana-1)/2);
    end
    
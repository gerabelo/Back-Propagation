/* 
        Rede Neural Artificial 
        geraldo.rabelo@gmail.com
        23/05/2017
	06/06/2017 - atualizacoes: funcoes de ativacao diversas; momentum.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define MAX_LINHA	1024
#define MAX_VALOR	10
#define DELIMITER	59
#define CONFIG		"config.dat"
#define LOG		"pesos_debug.csv"
#define TREINO          "treino.dat"
#define PREVISAO        "teste.dat"
#define LOGENABLE       0

struct inputMatrix 
{    
        float   *linha;
}       IMATRIX;
typedef struct inputMatrix InputMatrix;

struct targetMatrix 
{    
        float   *linha;
}       TMATRIX;
typedef struct targetMatrix TargetMatrix;

struct neuronio 
{    
        float   *pesosAntigos; //para o algoritmo BP
        float   *momentum; //para buscar o minimo global
        float   *entradas;
        float   *pesos;
        int     camada;
        float   saida;
}       RNA;
typedef struct neuronio Neuronio;

int     neuroniosQuantidade     = 0;
int     *neuroniosPorCamada     = NULL;
int     NUMERO_DE_CAMADAS       = 0;
float   taxaDeAprendizado       = 0.0; //loaded from file
int     tipoAtivacao            = 0;
int     MAX_EPOCAS              = 0;
int     DEBUG                   = 0;

void imprimeCreditos (void) {
	printf("\n\n\tRede Neural Artificial com Retro-Propagacao \n\tImplementado em 23/05/2017 por:\n\t\tGeraldo Rabelo (geraldo.rabelo@gmail.com)\n");
}

//how to use
void imprimeComoUsar (void) {
	printf("\n\tArquivos auxiliares:\n\t\ttreino.dat\t\tVetor de Entradas\n\t\tconfig.dat\t\tParametros de configuracao contendo:\n\n\t\t\t0.5\t\t\tTaxa de aprendizagem\n\t\t\t20000\t\t\tNumero Maximo de Epocas\n\t\t\t2\t\t\tFuncao de Ativacao: 1-Sigmoid 2-Tangente Hiperbolica 3-ArcTan 4-ReLU\n");

}

//how it works
void imprimeComoFunciona (void) {
	printf("\n\t\t\"Aprendizado Supervisionado por Correcao e Erro baseado no\n\t\tmetodo do Gradiente Descendente (minimizacao global, com Momentum)\"\n");
}

//função com 5 opções para ativação
float ativacao (float z, int tipo)
{
	float resultado = 0.0;

	switch(tipo)
	{
		case 1: 
			resultado = 1.0/(1.0 + exp(-z));			//sigmoid
			break;
		case 2: 
			resultado = (exp(z)-exp(-z))/(exp(z)+exp(-z));		//tanh
			break;
		case 3: 
			resultado = atan(z);					//arc tan
			break;
		case 4:
			if (z < 0) resultado = 0.01*z; else resultado = z;	//leaky ReLU
			break;
		default:
			resultado = exp(-pow(z,2));				//gaussian
			break;
	}

        return resultado;
}


float derivadaAtivacao (float z, int tipo)
{
	float resultado = 0.0;

	switch(tipo)
	{
		case 1: 
			resultado = z*(1-z);						//sigmoid
			break;
		case 2: 
			resultado = 1-pow(((exp(z)-exp(-z))/(exp(z)+exp(-z))),2);	//tanh
			break;
		case 3:
			resultado = 1/(pow(z,2)+1);					//arc tan
			break;
		case 4:
			if (z < 0) resultado = 0.01*z; else resultado = 1;		//leaky ReLU
			break;
		case 5: 
			resultado = -2*z*exp(-pow(z,2));				//Gaussian
			break;
	}

        return resultado;
}

Neuronio* criaRedeNeuronal(int *neuroniosPorCamada)
{
	int camadasQuantidade           = 0;
	int numeroDeEntradas            = 0;
	int neuronioAtual               = 0;

       	camadasQuantidade               = NUMERO_DE_CAMADAS;//1+sizeof(neuroniosPorCamada)/sizeof(neuroniosPorCamada[0]);
        
        int camadasQuantidade_contador  = camadasQuantidade;
        
        //contando o total de neuronios da rede
        while (camadasQuantidade_contador)
        {
                neuroniosQuantidade     = neuroniosQuantidade+neuroniosPorCamada[camadasQuantidade_contador-1];
                camadasQuantidade_contador--;
        }        
        
        //criando todos os neuronios de uma vez
        Neuronio *neuronios = malloc(neuroniosQuantidade*sizeof(Neuronio));
        
        //inicializando todos os neuronios por camadas
        for (int contadorCamadas = 0; contadorCamadas<camadasQuantidade; contadorCamadas++)
        {
                
                for (int contadorNeuroniosPorCamada = 0; contadorNeuroniosPorCamada<neuroniosPorCamada[contadorCamadas]; contadorNeuroniosPorCamada++)
                {

                        if (contadorCamadas == 0)
                        {
                                //neuronios da camada de entrada tem apenas uma entrada com peso unitário
                                neuronios[neuronioAtual].entradas       = (float *) malloc(sizeof(float));
                                neuronios[neuronioAtual].pesos          = (float *) malloc(sizeof(float));
				neuronios[neuronioAtual].pesosAntigos   = (float *) malloc(sizeof(float));
				neuronios[neuronioAtual].momentum       = (float *) malloc(sizeof(float));
                                
                                neuronios[neuronioAtual].entradas[0]    = 0.0; //vetor de entrada
                                neuronios[neuronioAtual].pesos[0]       = 1.0;
				neuronios[neuronioAtual].pesosAntigos[0]= neuronios[neuronioAtual].pesos[0];
                                neuronios[neuronioAtual].saida          = 0.0;
                                neuronios[neuronioAtual].camada         = 0;
				neuronios[neuronioAtual].momentum[0]    = 0.0;
                        }
                        else
                        {
                                //cada neuronio escondido terá quantidade de entradas igual à quantidade de neuronios na camada anterior
                                //multiplicar os sizeof pelo numero de neuronios da camada anterior
                                numeroDeEntradas = neuroniosPorCamada[contadorCamadas-1];
                                neuronios[neuronioAtual].entradas       = (float *) malloc(numeroDeEntradas*sizeof(float)); 
                                neuronios[neuronioAtual].pesos          = (float *) malloc(numeroDeEntradas*sizeof(float));
				neuronios[neuronioAtual].pesosAntigos   = (float *) malloc(numeroDeEntradas*sizeof(float));
				neuronios[neuronioAtual].momentum       = (float *) malloc(numeroDeEntradas*sizeof(float));
                                
                                for (int entrada = 0; entrada < numeroDeEntradas; entrada++)
                                {
                                        neuronios[neuronioAtual].entradas[entrada]      = 0.0;
                                        neuronios[neuronioAtual].pesos[entrada]         = (rand() % 10+1)/100.0;
					neuronios[neuronioAtual].pesosAntigos[entrada]  = neuronios[neuronioAtual].pesos[entrada];
					neuronios[neuronioAtual].momentum[entrada]      = 0.0;
                                }

                                neuronios[neuronioAtual].saida  = 0.0;
                                neuronios[neuronioAtual].camada = contadorCamadas;                        
                        }
                        if (neuronioAtual < neuroniosQuantidade) neuronioAtual++;
                }
        }
        return neuronios;
}


void ajustaPesosAntigos(Neuronio *neuronios,int *neuroniosPorCamada)
{
	int numeroDeCamadas             = NUMERO_DE_CAMADAS;//sizeof(neuroniosPorCamada)/sizeof(neuroniosPorCamada[0]);
	int numeroDeNeuronios           = 0;
	int numeroDePesos               = 0;

	for (int camada = 0; camada < numeroDeCamadas; camada++)	
	{
		numeroDeNeuronios += neuroniosPorCamada[camada];
	}

	for (int neuronio = 0; neuronio < numeroDeNeuronios; neuronio++)
	{
		numeroDePesos = sizeof(neuronios[neuronio].pesos)/sizeof(neuronios[neuronio].pesos[0]);
		for (int pesoContador = 0; pesoContador < numeroDePesos; pesoContador++)
		{
			neuronios[neuronio].pesosAntigos[pesoContador] = neuronios[neuronio].pesos[pesoContador];
		}		
	}
}

void imprimeRedeNeuronal(Neuronio *neuronios, int *neuroniosPorCamada)
{
        int camadasQuantidade   = NUMERO_DE_CAMADAS;//1+sizeof(neuroniosPorCamada)/sizeof(neuroniosPorCamada[0]);
        int neuronioAtual       = 0;
        int numeroDeEntradas    = 0;

	printf("\n\n[Imprime Rede Neural]:");

        for (int contadorCamadas = 0; contadorCamadas<camadasQuantidade; contadorCamadas++)
        {
                printf("\nTotal de neuronios na camada %d: %d",contadorCamadas,neuroniosPorCamada[contadorCamadas]);
                
                for (int contadorNeuroniosPorCamada = 0; contadorNeuroniosPorCamada<neuroniosPorCamada[contadorCamadas]; contadorNeuroniosPorCamada++)
                {
                        printf("\n Neuronio %d",neuronioAtual);
                        
                        if (contadorCamadas == 0)
                        {
                                printf("\n  Numero de entradas: 1");
                                //neuronios da camada de entrada tem apenas uma entrada com peso unitário                          
                                printf("\n   Valor da Entrada 1/Peso: %f / %f",neuronios[neuronioAtual].entradas[0],neuronios[neuronioAtual].pesos[0]);
                                printf("\n   Saida: %f",neuronios[neuronioAtual].saida);
                        }
                        else
                        {
                                numeroDeEntradas = neuroniosPorCamada[contadorCamadas-1];
                                printf("\n  Numero de entradas: %d",numeroDeEntradas);
                           
                                for (int entrada = 0; entrada < numeroDeEntradas; entrada++)
                                {
                                        printf("\n   Valor da Entrada %d/Peso: %f / %f",entrada,neuronios[neuronioAtual].entradas[entrada],neuronios[neuronioAtual].pesos[entrada]);
                                }
                                printf("\n    Saida: %f",neuronios[neuronioAtual].saida);
                        }
			if (neuronioAtual < neuroniosQuantidade) neuronioAtual++;
                }
        }
}


void imprimeResumo(Neuronio *neuronios, int *neuroniosPorCamada, FILE *arquivo)
//void imprimeResumo(Neuronio *neuronios, int *neuroniosPorCamada)
{
        int camadasQuantidade   = NUMERO_DE_CAMADAS;//1+sizeof(neuroniosPorCamada)/sizeof(neuroniosPorCamada[0]);
        int neuronioAtual       = 0;
        int numeroDeEntradas    = 0;
	int numeroDeNeuronios   = 0;

        for (int contadorCamadas = 0; contadorCamadas<camadasQuantidade; contadorCamadas++)
        {
                numeroDeNeuronios += neuroniosPorCamada[contadorCamadas];
        }

	if (DEBUG == 0) 
	{
	        printf("\n\t\t%d Neuronios distribuidos em %d camadas\n",numeroDeNeuronios,camadasQuantidade);

                for (int contadorCamadas = 0; contadorCamadas<camadasQuantidade; contadorCamadas++)
                {
                        printf("\t\t\t%d Neuronios na camada %d\n",neuroniosPorCamada[contadorCamadas],contadorCamadas);
                }
        }
        
        for (int contadorCamadas = 0; contadorCamadas<camadasQuantidade; contadorCamadas++)
        {
                if (DEBUG == 0) printf("\n\n\t\tPesos auto-ajustados para a camada %d:\n\t\t",contadorCamadas);

                for (int contadorNeuroniosPorCamada = 0; contadorNeuroniosPorCamada<neuroniosPorCamada[contadorCamadas]; contadorNeuroniosPorCamada++)
                {
                        if (contadorNeuroniosPorCamada % 4 == 0) printf("\n\t\t\t");
                        if (contadorCamadas == 0)
                        {
                                if (DEBUG == 0) printf("%f; ",neuronios[neuronioAtual].pesos[0]);
                                fprintf(arquivo,"%d:%d:%f\n",contadorCamadas,neuronioAtual,neuronios[neuronioAtual].pesos[0]);
                        }
                        else
                        {
                                numeroDeEntradas = neuroniosPorCamada[contadorCamadas-1];
                           
                                for (int entrada = 0; entrada < numeroDeEntradas; entrada++)
                                {
                                        if (DEBUG == 0) printf("%f; ",neuronios[neuronioAtual].pesos[entrada]);
                                        fprintf(arquivo,"%d:%d:%f\n",contadorCamadas,neuronioAtual,neuronios[neuronioAtual].pesos[entrada]);                                        
                                }
                        }
			if (neuronioAtual < neuroniosQuantidade) neuronioAtual++;
                }

        }
        if (DEBUG == 0) printf("\n\n\tPesos salvos em arquivo (\"output_*.dat\"), no formato CAMADA:NEURONIO:PESO");
}

float erroQuadratico (float saida, float alvo)
{
        return 0.5*pow((alvo-saida),2);
}

float derivadaDoErroQuadratico (float saida, float alvo)
{
	float resultado = saida-alvo;
	if (DEBUG == 1) printf("\n    derivada do Erro quadratico: %f - %f = %f",saida,alvo,resultado);
        return resultado;
}

int getUltimoNeuronioDaCamada (int camada, int *neuroniosPorCamada)
{
        int indexNeuronio = 0;

        for (int camadasContador = 0;camadasContador <= camada; camadasContador++)
        {
                indexNeuronio += neuroniosPorCamada[camadasContador];
        }
        
        return indexNeuronio-1;
}

int getPrimeiroNeuronioDaCamada (int camada, int *neuroniosPorCamada)
{
        int indexNeuronio = 0;

        for (int camadasContador = 0;camadasContador < camada; camadasContador++)
        {
                indexNeuronio += neuroniosPorCamada[camadasContador];
        }
        
        return indexNeuronio;
}

int getTotalDeNeuroniosNaRede (int *neuroniosPorCamada)
{
	int totalCamadas = NUMERO_DE_CAMADAS;
	int resultado = 0;

	for (int contador = 0; contador<totalCamadas; contador++)
	{
		resultado += neuroniosPorCamada[contador];		
	}	
	return resultado;
}

float calculaDelta (Neuronio *neuronios,int neuronioAtual,int *neuroniosPorCamada, int camadaAtual, float erro, int neuronioAnterior)
{
	int     quantidadeDeParcelasNaSomaDoDeltaAtual      = 0;
	int     numeroDeNeuroniosCamadaSeguinte             = 0;
	int     indexNeuroniosCamadaSeguinte                = 0;
	int     contadorDeslocadoPeloBias                   = 0;
	int     posicaoNeuronioCamada                       = 0;
	int     indexNeuronio                               = 0;
	int     totalCamadas                                = NUMERO_DE_CAMADAS;
	float   resultado                                   = 0.0;
	float   delta                                       = 0.0;
	
	quantidadeDeParcelasNaSomaDoDeltaAtual          = numeroDeNeuroniosCamadaSeguinte;
	numeroDeNeuroniosCamadaSeguinte                 = neuroniosPorCamada[camadaAtual+1];

	if (DEBUG == 1) printf("\n   [Delta]:\n   Camada %d de %d",camadaAtual,totalCamadas);
	
	if (camadaAtual < (totalCamadas-1))
	{
		//camada escondida
		if (quantidadeDeParcelasNaSomaDoDeltaAtual == 1)
		{

        		indexNeuroniosCamadaSeguinte = getTotalDeNeuroniosNaRede(neuroniosPorCamada)-neuroniosPorCamada[camadaAtual+1];	
			if (DEBUG == 1) printf("\n   Neuronio %d; Peso %d; Parcela %d de %d",indexNeuroniosCamadaSeguinte,contadorDeslocadoPeloBias,neuronioAtual,quantidadeDeParcelasNaSomaDoDeltaAtual);

			resultado += neuronios[indexNeuroniosCamadaSeguinte].pesosAntigos[neuronioAtual-2]*calculaDelta(neuronios,indexNeuroniosCamadaSeguinte,neuroniosPorCamada,camadaAtual+1,erro,neuronioAtual);

		} else {
			for (int contador = 0; contador < quantidadeDeParcelasNaSomaDoDeltaAtual; contador++)
			{
				indexNeuroniosCamadaSeguinte = getTotalDeNeuroniosNaRede(neuroniosPorCamada)-neuroniosPorCamada[camadaAtual+1]+contador;	
				contadorDeslocadoPeloBias = contador+0;
				if (DEBUG == 1) printf("\n   Neuronio %d; Peso %d; Parcela %d de %d",indexNeuroniosCamadaSeguinte,contadorDeslocadoPeloBias,contador,quantidadeDeParcelasNaSomaDoDeltaAtual);

				resultado += neuronios[indexNeuroniosCamadaSeguinte].pesosAntigos[contadorDeslocadoPeloBias]*calculaDelta(neuronios,indexNeuroniosCamadaSeguinte,neuroniosPorCamada,camadaAtual+1,erro,neuronioAtual);
			}
		}
	} else {
		//ultima camada
		//alterar isto para multiplas saidas
		if (DEBUG == 1) printf("\n   Neuronio %d; Parcela %d; Camada %d de %d",neuronioAtual,quantidadeDeParcelasNaSomaDoDeltaAtual,camadaAtual,totalCamadas);
		resultado = erro;		
	}

	if (DEBUG == 1) printf("\n    Delta Resultado: %f",resultado);
	return resultado;

}

float soma (int camadaAnterior, int *neuroniosPorCamada, int neuronio, Neuronio *neuronios)
{
        int     totalNeuroniosCamadasAnteriores         = 0;
        int     contadorNeuronioCamadaAnterior          = 0;
        int     primeiroNeuronioCamadaAnterior          = 0;
        int     camadaAnteriorContador                  = camadaAnterior;
	float   saidaDoNeuronioAtual                    = 0.0;
        float   produtoParcial                          = 0.0;
        float   saidaAnterior                           = 0.0;
        float   entradaAtual                            = 0.0;
        float   resultado                               = 0.0;
        float   pesoAtual                               = 0.0;
        
        primeiroNeuronioCamadaAnterior = getPrimeiroNeuronioDaCamada (camadaAnterior,neuroniosPorCamada);
        if (DEBUG == 1) printf("\n\n [SOMA]\n Total neuronios na camada %d: %d; primeiro: %d",camadaAnterior,neuroniosPorCamada[camadaAnterior],primeiroNeuronioCamadaAnterior);
        int linhaContador = 0;
        
//      BIAS
//        resultado = neuronios[neuronio].entradas[0]*neuronios[neuronio].pesos[0];
//        if (DEBUG == 1) printf("\n  [Bias] entrada: %f; peso: %f",neuronios[neuronio].entradas[0],neuronios[neuronio].pesos[0]);
        
        for (int contadorNeuroniosPorCamada = primeiroNeuronioCamadaAnterior; contadorNeuroniosPorCamada < (neuroniosPorCamada[camadaAnterior]+primeiroNeuronioCamadaAnterior);contadorNeuroniosPorCamada++)
        {
                saidaAnterior   = neuronios[contadorNeuroniosPorCamada].saida;
                pesoAtual       = neuronios[neuronio].pesos[linhaContador];
                produtoParcial  = saidaAnterior*pesoAtual;
                resultado       = resultado+produtoParcial;

                neuronios[neuronio].entradas[linhaContador]     = produtoParcial;
                entradaAtual                                    = neuronios[neuronio].entradas[linhaContador];
                
                if (DEBUG == 1) printf("\n -- [Neuronio %d <- %d:%f] Valor na Entrada %d do neuronio %d: %f; Peso: %f; Resultado: %f",neuronio,contadorNeuroniosPorCamada,neuronios[contadorNeuroniosPorCamada].saida,linhaContador,neuronio,entradaAtual,pesoAtual,resultado);
                linhaContador++;
        }
        
        return resultado;
}

int backpropagation(float saida, float erro, int *neuroniosPorCamada, Neuronio *neuronios)
{
	float   saidaDoNeuronioAnterior = 0.0;
	float   saidaDoNeuronioAtual    = 0.0;
        int     ultimoNeuronio          = 0;
        int     inputsNeuronio          = 0;
        int     indexNeuronio           = 0;
	float   derivadaSaida           = 0.0;
        int     totalCamadas            = NUMERO_DE_CAMADAS;
	float   taxaMomentum            = 0.0;
        int     camadaAtual             = 0;
	float   pesoAntigo              = 0.0;
	float   deltaPeso               = 0.0;
	float   derivada                = 0.0;
	float   novoPeso                = 0.0;
	float   momentum                = 0.0;
	float   delta                   = 0.0;
        
        if (DEBUG == 1) printf("\n\n[Back Propagation] :");
        
        while (totalCamadas > 0)
        {
                //a camada de entrada (0) nao tem INPUTS a serem ajustados
                //portanto deve saltar para a primeira camada oculta
                
                camadaAtual     = totalCamadas;
                if (DEBUG == 1) printf("\n\n Camada atual: %d",camadaAtual),printf("\n Neuronios nesta camada: %d",neuroniosPorCamada[camadaAtual]); 
                
                //pegar o primeiro neuronio desta camada
                indexNeuronio = getPrimeiroNeuronioDaCamada(camadaAtual,neuroniosPorCamada);
                //pegar o ultimo neuronio desta camada
                ultimoNeuronio  = getUltimoNeuronioDaCamada(camadaAtual,neuroniosPorCamada);
                if (DEBUG == 1) printf("\n primeiro Neuronio: %d; ultimo Neuronio: %d",indexNeuronio,ultimoNeuronio);
                
                //processa neuronios desda camada
                for (int neuronioContador = indexNeuronio; neuronioContador <= ultimoNeuronio; neuronioContador++)
                {
                        inputsNeuronio = neuroniosPorCamada[camadaAtual-1];
//                        delta = calculaDelta(neuronios,neuronioContador);

                        if (DEBUG == 1)
                        {
                                printf("\n Entradas do neuronio %d: %d",neuronioContador,inputsNeuronio);
                                //processar as entradas do neuronio
                                for (int entradaContador = 0; entradaContador < inputsNeuronio; entradaContador++)
                                {
                                        printf("\n  (%d) valor: %f", entradaContador,neuronios[neuronioContador].entradas[entradaContador]);
                                }
                        }

			//numero de pesos é igual ao numero de inputs
			//pesos[0] é bias. fixo em 1.
			//o algoritmo do BP pode reduzir o valor do bias (inicialmente grande) enquanto aumenta o valor dos pesos.
			if (DEBUG == 1) printf("\n Pesos (sem ajuste) do neuronio %d: %d",neuronioContador,inputsNeuronio);
                        for (int pesosContador = 0; pesosContador < inputsNeuronio; pesosContador++)
                        {
                                if (DEBUG == 1) printf("\n  (%d) valor: %f", pesosContador,neuronios[neuronioContador].pesos[pesosContador]);
				//if (pesosContador > 0)
				//{
					saidaDoNeuronioAnterior = neuronios[neuronioContador].entradas[pesosContador]/neuronios[neuronioContador].pesos[pesosContador];
					saidaDoNeuronioAtual    = neuronios[neuronioContador].saida;
					delta                   = calculaDelta(neuronios,neuronioContador,neuroniosPorCamada,camadaAtual,erro,neuronioContador);
					derivada                = derivadaAtivacao(saidaDoNeuronioAtual,tipoAtivacao);
					derivadaSaida           = derivadaAtivacao(saida,tipoAtivacao);
					pesoAntigo              = neuronios[neuronioContador].pesos[pesosContador];

					if (derivada != derivadaSaida)
					{
						taxaMomentum = taxaDeAprendizado+((rand() % 10+1)/100.0);
						//printf("\ntaxaMomento: %f",taxaMomentum);
						deltaPeso = delta*derivada*derivadaSaida*saidaDoNeuronioAnterior;

						if (neuronios[neuronioContador].momentum[pesosContador] == 0.0)
						{
							novoPeso = pesoAntigo - taxaDeAprendizado*deltaPeso;
						} else {
							momentum = neuronios[neuronioContador].momentum[pesosContador];
							novoPeso = pesoAntigo - taxaDeAprendizado*(deltaPeso-(taxaMomentum*momentum));
						}

						neuronios[neuronioContador].momentum[pesosContador] = deltaPeso;

						if (DEBUG == 1) printf("\n  (%d) Ajustado para: %f >> %f + %f*%f*%f*%f*%f; DerivadaSaida: %f",pesosContador,novoPeso,pesoAntigo,taxaDeAprendizado,delta,derivada,saidaDoNeuronioAnterior,derivadaSaida,derivadaSaida);

					} else {

						taxaMomentum = taxaDeAprendizado+((rand() % 10+1)/100.0);
						deltaPeso = delta*derivada*saidaDoNeuronioAnterior;
						//printf("\ntaxaMomento: %f",taxaMomentum);

						if (neuronios[neuronioContador].momentum[pesosContador] == 0.0)
						{
							novoPeso = pesoAntigo - taxaDeAprendizado*deltaPeso;
						} else {
							momentum = neuronios[neuronioContador].momentum[pesosContador];
							novoPeso = pesoAntigo - taxaDeAprendizado*(deltaPeso-(taxaMomentum*momentum));
						}

						neuronios[neuronioContador].momentum[pesosContador] = deltaPeso;

						if (DEBUG == 1) printf("\n  (%d) Ajustado para: %f >> %f + %f*%f*%f*%f",pesosContador,novoPeso,pesoAntigo,taxaDeAprendizado,delta,derivada,saidaDoNeuronioAnterior);
					}
					neuronios[neuronioContador].pesosAntigos[pesosContador] = neuronios[neuronioContador].pesos[pesosContador];
					neuronios[neuronioContador].pesos[pesosContador] = novoPeso;
				//}
                        }
                }
                totalCamadas--;
        }

        return 1;
}

void logaPesos(FILE *log, Neuronio *neuronios,int *neuroniosPorCamada, float erro)
{
	int totalPesosDoNeuronio        = 0;
        int contadorNeuronios           = 0;
        int contadorCamadas             = 0;
	int totalNeuronios              = 0;
        int neuronioAtual               = 0;
	int totalCamadas                = NUMERO_DE_CAMADAS;
	int pesoContador                = 0;
        int camadaAtual                 = 0;

	fprintf(log,"%f",erro);

	for(contadorCamadas = 0; contadorCamadas <= totalCamadas; contadorCamadas++)
	{
		totalNeuronios += neuroniosPorCamada[contadorCamadas];
	}
        
	for (contadorCamadas = 0; contadorCamadas <= totalCamadas; contadorCamadas++)
	{
	        for (contadorNeuronios = 0; contadorNeuronios < neuroniosPorCamada[contadorCamadas]; contadorNeuronios++)
	        {        
                        if (contadorCamadas == 0)
                        {
                                totalPesosDoNeuronio = 1; 
                        } else {
                                totalPesosDoNeuronio = neuroniosPorCamada[contadorCamadas-1];
                        }	        
	        	        
		        for (pesoContador = 0; pesoContador < totalPesosDoNeuronio; pesoContador++)
		        {
			        fprintf(log,",%f",neuronios[neuronioAtual].pesos[pesoContador]);
		        }
		        neuronioAtual++;
	        }		
	}	
}

char *getDescricaoTipoAtivacao(int tipo)
{
        switch(tipo)
	{
		case 1: 
			return "SIGMOID";
			break;
		case 2: 
			return "TANH";
			break;
		case 3: 
			return "ATAN";
			break;
		case 4:
			return "Leaky ReLU";
			break;
		case 5:
			return "GAUSSIAN";
			break;
	}
}

int treinaRedeNeuronal (Neuronio *neuronios, int *neuroniosPorCamada, InputMatrix *inputs, TargetMatrix *targets, FILE *log)
{
        int     neuronioPorCamadaContador       = 0;
        int     entradaCicloContador            = 0;        
        int     entradasQuantidade              = -1;
        int     camadasQuantidade               = NUMERO_DE_CAMADAS;
        int     saidasQuantidade                = 0;
        int     neuronioContador                = 0;
        int     entradaContador                 = 0;
        int     targetContador                  = 0;
        int     camadaContador                  = 0;
        int     epocaContador                   = 0;
        int     camadaAtual                     = 0;
	float   somaTotal                       = 0.0;
        int     entrada                         = 0;
        int     target                          = 0;
	float   saida                           = 0.0;
        float   erro                            = 0.0;
        
        InputMatrix *entradas                   = inputs;
        TargetMatrix *alvos                     = targets;
        
        saidasQuantidade                        = neuroniosPorCamada[camadasQuantidade-1];
                
        for (epocaContador = 0;epocaContador < MAX_EPOCAS; epocaContador++)
        {
                if (DEBUG == 2) printf("\n\nEpoca %d de %d:",epocaContador+1,MAX_EPOCAS);

                entradasQuantidade      = 0;
                neuronioContador        = 0;
                entradaContador         = 0;
                
                while (entradas[entradaContador].linha)
                {
                  entradaContador++;
                  entradasQuantidade++;
                }

                if (DEBUG == 5) printf("\n%d entradas",entradasQuantidade);

                while (entradasQuantidade > 0)
                {
                        if (DEBUG == 2) { printf("\n"); }
                        entradasQuantidade--;                
                        neuronioContador        = 0;
                        camadaAtual             = 0;

			if (DEBUG == 1) imprimeRedeNeuronal(neuronios,neuroniosPorCamada);

                        while (camadaAtual < camadasQuantidade)
                        {
                                if (camadaAtual == 0)
                                {
                                        neuronioPorCamadaContador = 0;
                                        
                                        while (neuronioPorCamadaContador < neuroniosPorCamada[camadaAtual])
                                        {
                                                entradaContador = neuronioPorCamadaContador;
                                                if (DEBUG == 1) printf("\n>> Numero de Neuronios da Camada %d: %d",camadaAtual,neuroniosPorCamada[camadaAtual]);
                                                if (DEBUG == 5) printf("\n%d dimensões",neuroniosPorCamada[0]);
                                                
                                                while (entradaContador < neuroniosPorCamada[camadaAtual])
                                                {
                                                        
                                                        neuronios[neuronioPorCamadaContador].saida = entradas[entradasQuantidade].linha[entradaContador];
                                                        if ((DEBUG == 1) || (DEBUG == 5)) printf("\n [Neuronio %d] Saida: %f; Entrada contador: %d",neuronioPorCamadaContador,neuronios[neuronioPorCamadaContador].saida,entradaContador);
                                                        neuronioContador++;
                                                        neuronioPorCamadaContador++;
                                                        entradaContador++;
                                                }
                                        }
                                } else {
                                                entradaContador = 0;
                                                if (DEBUG == 1) printf("\n\nNumero de Neuronios da Camada %d: %d",camadaAtual,neuroniosPorCamada[camadaAtual]);
                                                while (entradaContador < neuroniosPorCamada[camadaAtual])
                                                {                                                        
                                                        somaTotal = soma(camadaAtual-1,neuroniosPorCamada,neuronioContador,neuronios);
							saida = ativacao(somaTotal,tipoAtivacao);
							if (DEBUG == 1) printf("\n -- Soma: %f; Ativacao: %f",somaTotal,saida);
                                                        neuronios[neuronioContador].saida = saida;
                                                        if ((DEBUG == 1) || (DEBUG == 5)) 
                                                        {
                                                                printf("\n -- [Neuronio: %d] Saida: %f; Index neuronio-camada: %d",neuronioContador,neuronios[neuronioContador].saida,entradaContador);
							        for (int entrada = 0; entrada < neuroniosPorCamada[camadaAtual-1]; entrada++) 
							        {
								        printf("\n ---  Valor na entrada %d do neuronio %d: %f",entrada,neuronioContador,neuronios[neuronioContador].entradas[entrada]);
							        }
                                                        }
                                                        neuronioContador++;                                                        
                                                        entradaContador++;
                                                }
                                                //somar as saídas da camada anterior vezes os pesos da entrada da camada atual
                                }
                                camadaAtual++;
                        }

                        if ((DEBUG == 2) || (DEBUG == 5)) printf("\n ---  [Linha %d] Saida: %f, Alvo: %f",entradasQuantidade,neuronios[neuronioContador-1].saida,targets[0].linha[entradasQuantidade]);
			if (DEBUG == 0) if (epocaContador == MAX_EPOCAS-1) 
			{
			        printf("\n\t\tEntrada %d: { ",entradasQuantidade);
			        for (entradaContador = 0; entradaContador < neuroniosPorCamada[0]-1; entradaContador++)
			        {
			                printf("%f,",neuronios[entradaContador].saida);
			        }
			        
			        printf("%f }; Alvo: %f; Previsão: %f",neuronios[entradaContador].saida,targets[0].linha[entradasQuantidade],neuronios[neuronioContador-1].saida);
			}
                        //printf(", Erro Quadratico: %f",erro);
                        erro = derivadaDoErroQuadratico(neuronios[neuronioContador-1].saida,targets[0].linha[entradasQuantidade]);

		if (LOGENABLE) {
		        logaPesos(log,neuronios,neuroniosPorCamada,erroQuadratico(neuronios[neuronioContador-1].saida,targets[0].linha[entradasQuantidade]));
        		fprintf(log,"\n");
		}
                        if (!backpropagation(neuronios[neuronioContador-1].saida,erro,neuroniosPorCamada,neuronios)) { exit(1); };
			ajustaPesosAntigos(neuronios,neuroniosPorCamada);
                }
        }

	if (DEBUG == 0) printf ("\n\n\t\tNumero de Epocas: %d",MAX_EPOCAS);
	if (DEBUG == 0) printf ("\n\t\tTaxa de Aprendizado: %f\n",taxaDeAprendizado);
	if (DEBUG == 0)
	{
		printf("\n\t\tFuncao de Ativacao: (%d) ",tipoAtivacao);
					
		char retorno[11];
		memset(retorno,'\0',11);
		sprintf(retorno,"%s",getDescricaoTipoAtivacao(tipoAtivacao));
					
		printf("%s",retorno);
                printf("\n");
	}
        return 1;
}

int getLinhasArquivo(FILE *arquivo)
{
	int     resultado = 0;
	char    caractere;
	
	rewind(arquivo);
	while (!feof(arquivo))
	{
		caractere = fgetc(arquivo);
		if(caractere == '\n')
		{
		  resultado++;
		}
	}
	return resultado;
}

int getEntradasPorLinha(FILE *arquivo)
{
	int     resultado = 0;
	char    caractere;
	
	rewind(arquivo);
	while (!feof(arquivo))
	{
		caractere = fgetc(arquivo);
		if (caractere == ';') resultado++;
		if (caractere == '\n') break;
	}
	return resultado;
}

char *substring (char *linha, int inicio, int fim)
{
	char *resultado = (char *) malloc(strlen(linha)*sizeof(char));

	for (int contador = inicio; contador<fim; contador++)
	{
		resultado[contador-inicio] = linha[contador];

	}
	
	return resultado;
} 

float getAlvos (FILE *arquivo, int linhaArquivo)
{
        float   resultado               = 0.0;
	int     linhaContador           = 0;
	int     entradaContador         = 0;
	int     valorContador           = 0;
	char    caractere;
	char    linha[MAX_LINHA];
	char    valor[MAX_VALOR];
	
	rewind(arquivo);
	while (!feof(arquivo))
	{
		fgets(linha,MAX_LINHA,arquivo);
		if (linhaContador == linhaArquivo)
		{
			valorContador = 0;
			for (int contador=0; contador < strlen(linha); contador++)
			{
				if (linha[contador] == ':')
				{
					resultado = atof(substring(linha,(contador+1),strlen(linha)));
				}
			}
		}
		memset(linha,'\0',MAX_LINHA);
		linhaContador++;
	}
        
        return resultado;
}

float getEntrada(FILE *arquivo, int linhaArquivo, int entradaArquivo)
{
	int     entradaContador         = 0;
	int     valorContador           = 0;
	int     linhaContador           = 0;
	float   resultado               = 0.0;
	char    caractere;
	char    linha[MAX_LINHA];
	char    valor[MAX_VALOR];
	
	rewind(arquivo);
	while (!feof(arquivo))
	{
		fgets(linha,MAX_LINHA,arquivo);
		if (linhaContador == linhaArquivo)
		{
			valorContador = 0;
			for (int contador=0; contador < strlen(linha); contador++)
			{
				if (linha[contador] != ';')
				{
					valor[valorContador] = linha[contador];
					valorContador++;
				} else {
					if (entradaContador == entradaArquivo)
					{
						resultado = atof(valor);
						//break;
						return resultado;
					}
					valorContador = 0;
					memset(valor,'\0',MAX_VALOR);
					entradaContador++;
				}
			}
		}
		memset(linha,'\0',MAX_LINHA);
		linhaContador++;
	}
	return resultado;
}



void lerNeuroniosNaLinha(char *linha ,int *neuroniosPorCamada)
{
	int neuronioContador    = 0;
	int valor               = 0;
	int inicio              = 0;

	for (int contador = 0; contador < strlen(linha);contador++)
	{
		if (linha[contador] == ';')
		{
			valor = atoi(substring(linha,inicio+1,contador));
			neuroniosPorCamada[neuronioContador] = valor;
			neuronioContador++;
			inicio = contador;
		}
	}
}


void carregaConfig(FILE *arquivo)
{
        int     linhaContador = 0;
        char    valor[8];
        char    linha[32];

        memset(linha,'\0',32);
        memset(valor,'\0',8);

	rewind(arquivo);

	while (!feof(arquivo))
	{
		fgets(linha,32,arquivo);
	        for (int contador = 0; contador < strlen(linha); contador++)
                {
                        if (linha[contador] == ':')
                        {
                                for (int contadorValorCaracteres = contador+1; contadorValorCaracteres < strlen(linha); contadorValorCaracteres++)
                                {
                                        valor[contadorValorCaracteres-contador-1] = linha[contadorValorCaracteres];
                                }
                                if (linhaContador == 0) { MAX_EPOCAS = atoi(valor); }
                                if (linhaContador == 1) { taxaDeAprendizado = atof(valor); }
                                if (linhaContador == 2) { DEBUG = atoi(valor); }
                                if (linhaContador == 3) { tipoAtivacao = atoi(valor); }
                                if (linhaContador == 4)
				{
					NUMERO_DE_CAMADAS = atoi(valor);
					neuroniosPorCamada = malloc(NUMERO_DE_CAMADAS*sizeof(int));
				}
                                if (linhaContador == 5) {
					lerNeuroniosNaLinha(linha,neuroniosPorCamada);	
				}

                        }
                }		
                memset(linha,'\0',32);
                memset(valor,'\0',8);
		linhaContador++;
	}
}

void prever (Neuronio *neuronios,int *neuroniosPorCamada, InputMatrix *previsao, float *previsaoResultados)
{
	int previsaoResultadoContador   = 0;
        int contadorNeuronioAnterior    = 0;
        int primeiroNeuronioDaCamada    = 0;
        int totalNeuroniosNaCamada      = 0;
        int totalLinhasDeEntrada        = 0;
        int neuronioContador            = 0;
        int totalNeuronios              = 0;        
        int neuronioAtual               = 0;
        int totalEntradas               = neuroniosPorCamada[0];
        int totalCamadas                = NUMERO_DE_CAMADAS;
        float soma                      = 0.0;
       
        while (previsao[totalLinhasDeEntrada].linha)
        {
                totalLinhasDeEntrada++;
        }

        for (int contadorCamadas = 0; contadorCamadas < totalCamadas; contadorCamadas++)
        {
                totalNeuronios += neuroniosPorCamada[contadorCamadas];
        }
        
        for (int contadorLinhasEntrada = 0; contadorLinhasEntrada < totalLinhasDeEntrada; contadorLinhasEntrada++)
        {
                for (int contadorEntradas = 0; contadorEntradas<totalEntradas; contadorEntradas++)
                {
                        neuronios[contadorEntradas].saida = previsao[contadorLinhasEntrada].linha[contadorEntradas];
                }
                
                
                for (int contadorCamadas = 1; contadorCamadas < totalCamadas; contadorCamadas++)
                {
                        totalNeuroniosNaCamada = neuroniosPorCamada[contadorCamadas];
                        primeiroNeuronioDaCamada = getPrimeiroNeuronioDaCamada(contadorCamadas,neuroniosPorCamada);

                        for (neuronioContador = primeiroNeuronioDaCamada; neuronioContador < (totalNeuroniosNaCamada+primeiroNeuronioDaCamada); neuronioContador++)
                        {
                                soma = 0.0;
                                totalEntradas = neuroniosPorCamada[contadorCamadas-1];
                                contadorNeuronioAnterior = getPrimeiroNeuronioDaCamada(contadorCamadas-1,neuroniosPorCamada);
                                
                                for (int contadorEntradas = 0; contadorEntradas < totalEntradas; contadorEntradas++)
                                {
                                        neuronios[neuronioContador].entradas[contadorEntradas] = neuronios[neuronioContador].pesos[contadorEntradas]*neuronios[contadorNeuronioAnterior].saida;
                                        
                                        soma += neuronios[neuronioContador].entradas[contadorEntradas];
                                        contadorNeuronioAnterior++;
                                }                        
                                neuronios[neuronioContador].saida = ativacao(soma,tipoAtivacao);
                        }              
                }
                previsaoResultados[previsaoResultadoContador] = neuronios[neuronioContador-1].saida;
                previsaoResultadoContador++;
        }
}

void imprimePrevisao(Neuronio *neuronios,InputMatrix *previsao, FILE *testeFile, float *previsaoResultados)
{
        int quantidadeDeLinhasDeEntrada = getLinhasArquivo(testeFile);
        int quantidadeEntradasPorLinha  = getEntradasPorLinha(testeFile);
                
        printf("\n\n\t[ Base de Teste: \"%s\" ]\n\n\t\tEntradas -> Previsões\n\n",PREVISAO);
	for (int contadorLinha = 0; contadorLinha < quantidadeDeLinhasDeEntrada; contadorLinha++)
	{
	        printf("\t\t");
		for (int contadorEntrada = 0; contadorEntrada < quantidadeEntradasPorLinha; contadorEntrada++)
		{
			printf("%f; ",previsao[contadorLinha].linha[contadorEntrada]);
		}
		printf("\t->\t%f\n",previsaoResultados[contadorLinha]);
	}
}

int main(void)
{
        char saida[32];
        memset(saida,'\0',32);

	FILE *treinoFile        = NULL;
	FILE *configFile        = NULL;
        FILE *testeFile         = NULL;
	FILE *saidaFile         = NULL;
	FILE *logFile           = NULL;

	treinoFile              = fopen(TREINO,"r");
	configFile              = fopen(CONFIG,"r");
	testeFile               = fopen(PREVISAO,"r");
	logFile                 = fopen(LOG,"w+");

	if (treinoFile == NULL || configFile == NULL)
	{
	        printf("\nVerifique os arquivos .dat\n");
		exit(1);
	}
	
	int quantidadeDeLinhasDeEntrada = getLinhasArquivo(treinoFile);
	int quantidadeEntradasPorLinha  = getEntradasPorLinha(treinoFile);	
	int quantidadeAlvosPorLinha     = quantidadeEntradasPorLinha;
	
        InputMatrix *inputs             = malloc(quantidadeDeLinhasDeEntrada*sizeof(InputMatrix));
        
	for (int contadorLinha = 0; contadorLinha < quantidadeDeLinhasDeEntrada; contadorLinha++)
	{
		inputs[contadorLinha].linha = malloc(quantidadeEntradasPorLinha*sizeof(float));
		for (int contadorEntrada = 0; contadorEntrada < quantidadeEntradasPorLinha; contadorEntrada++)
		{
			inputs[contadorLinha].linha[contadorEntrada] = getEntrada(treinoFile,contadorLinha,contadorEntrada);
		}
	}
	
	
        TargetMatrix *targets           = malloc(sizeof(TargetMatrix));
        targets[0].linha                = (float *) malloc(quantidadeDeLinhasDeEntrada*sizeof(float));
        
        for (int contadorLinha = 0; contadorLinha < quantidadeDeLinhasDeEntrada; contadorLinha++)
	{
		targets[0].linha[contadorLinha] = getAlvos(treinoFile,contadorLinha);
	}
	
	carregaConfig(configFile);
	
        sprintf(saida,"output_%s_A%.2f_E%d_N%d_C%d.dat",getDescricaoTipoAtivacao(tipoAtivacao),taxaDeAprendizado,MAX_EPOCAS,getTotalDeNeuroniosNaRede(neuroniosPorCamada),NUMERO_DE_CAMADAS);

        printf("\n%s\n",saida);

        Neuronio *RNA                   = criaRedeNeuronal(neuroniosPorCamada);
        saidaFile                       = fopen(saida,"w+");

	imprimeCreditos();
	imprimeComoFunciona();
	imprimeComoUsar();
	printf("\n\n\t[ Resultado do Processo de Aprendizado ]\n");
	treinaRedeNeuronal(RNA,neuroniosPorCamada,inputs,targets,logFile);
	imprimeResumo(RNA,neuroniosPorCamada,saidaFile);
        printf("\n\n");
                       	
	if (testeFile != NULL)
	{
                quantidadeDeLinhasDeEntrada     = getLinhasArquivo(testeFile);
                quantidadeEntradasPorLinha      = getEntradasPorLinha(testeFile);
                
                InputMatrix *previsao           = malloc(quantidadeDeLinhasDeEntrada*sizeof(InputMatrix));
                float *previsaoResultados       = malloc(quantidadeDeLinhasDeEntrada*sizeof(InputMatrix));
                
	        for (int contadorLinha = 0; contadorLinha < quantidadeDeLinhasDeEntrada; contadorLinha++)
	        {
		        previsao[contadorLinha].linha = malloc(quantidadeEntradasPorLinha*sizeof(float));
		        for (int contadorEntrada = 0; contadorEntrada < quantidadeEntradasPorLinha; contadorEntrada++)
		        {
			        previsao[contadorLinha].linha[contadorEntrada] = getEntrada(testeFile,contadorLinha,contadorEntrada);
		        }
	        }

	        prever(RNA,neuroniosPorCamada,previsao,previsaoResultados);
	        imprimePrevisao(RNA,previsao,testeFile,previsaoResultados);
	        fclose(testeFile);
        }
               
	fclose(treinoFile);
	fclose(configFile);
	fclose(saidaFile);
	fclose(logFile);
	
	printf("\n\n");	
        return 1;       
}

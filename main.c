/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Multilayer Perceptron implementation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

//#include "mlp_trainer.h"
#include "mlp_classifier.h"
//#include "read_csv.h"
#include "simpleserial.h"
#include "hal.h"

parameters* param;
int* layer_sizes;

// uint8_t mlp_classifier_wrapper(uint8_t* arg1, uint8_t arg2) {
//     // Call the mlp_classifier function with the global variables

//     mlp_classifier(param, layer_sizes);

//     // Return a dummy value (assuming simpleserial_addcmd doesn't need a meaningful return value)
//     return 0;
// }

uint8_t mlp(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *in) {
    /* Device reset detected */
    //putch('r');
    //printf("Device reset detected\n");
    // putch('R');
    // putch('E');
    // putch('S');
    // putch('E');
    // putch('T');
    // putch(' ');
    // putch(' ');
    // putch(' ');
    // putch('\n');


    //HARDCODED VALS
    char *hardcoded_args[] = {
        "executable_name", // new_argv[0] is typically the program name
        "3",                // Number of hidden layers
        "4,5,5",            // Size of each hidden layer
        "softmax,relu,tanh",
        "1",           
        "sigmoid",         
        "0.01",              
        "10000 ",     
        "data/data_train.csv", 
        "1096",             
        "5",               
        "data/data_test.csv",
        "275",              
        "5",                
        NULL                // Sentinel to mark the end of the array
    };

    int new_argc = sizeof(hardcoded_args) / sizeof(char *) - 1; // Exclude the NULL sentinel

    // Allocate memory for the new new_argv
    char **new_argv = (char **)malloc(new_argc * sizeof(char *));

    // Copy the hardcoded strings to the new new_argv
    for (int i = 0; i < new_argc; i++) {
        new_argv[i] = (char *)malloc((strlen(hardcoded_args[i]) + 1) * sizeof(char));
        strcpy(new_argv[i], hardcoded_args[i]);
    }

    /*
    new_argv[0]: Executable file name Ex: a.out
    new_argv[1]: Number of hidden layers Ex: 3
    new_argv[2]: Size of each hidden layer separated by comma Ex: 4,5,5
    new_argv[3]: Hidden activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    new_argv[4]: Alpha (L2 Regularization parameter value)
    new_argv[5]: Maximum number of iterations
    new_argv[6]: Number of units in output layer
    new_argv[7]: Output activation function (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    new_argv[8]: Name of the csv file containing the train dataset
    new_argv[9]: Number of rows or samples in the train dataset
    new_argv[10]: Number of features including the output variable in the train dataset
    new_argv[11]: Name of the csv file containing the test dataset
    new_argv[12]: Number of rows or samples in the test dataset
    new_argv[13]: Number of features including the output variable in the test dataset
    */

    // Sanity check of command line arguments
    // if (argc != 14) {
    //     // Print help for execution syntax
    //     printf("\nExecution syntax:\n");
    //     printf("-----------------\n");
    //     printf("Argument 0: Executable file name Ex: ./MLP \n");
    //     printf("Argument 1: Number of hidden layers Ex: 3 \n");
    //     printf("Argument 2: Number of units in each hidden layer from left to right separated by comma (no spaces in-between) Ex: 4,5,5 \n");
    //     printf("Argument 3: Activation function of each hidden layer from left to right separated by comma (no spaces in-between) Ex: softmax,relu,tanh \n");
    //     printf("Argument 4: Number of units in output layer (Specify 1 for binary classification and k for k-class multi-class classification) Ex: 1 \n");
    //     printf("Argument 5: Output activation function Ex: sigmoid \n");
    //     printf("Argument 6: Learning rate parameter Ex: 0.01 \n");
    //     printf("Argument 7: Maximum number of iterations to run during training Ex: 10000 \n");
    //     printf("Argument 8: Path of the csv file containing the train dataset Ex: data/data_train.csv \n");
    //     printf("Argument 9: Number of rows in the train dataset (Number of samples) Ex: 1096 \n");
    //     printf("Argument 10: Number of columns in the train dataset (Number of input features + 1 (output variable)). The output variable should always be in the last column Ex: 5 \n");
    //     printf("Argument 11: Path of the csv file containing the test dataset Ex: data/data_test.csv \n");
    //     printf("Argument 12: Number of rows in the test dataset (Number of samples) Ex: 275 \n");
    //     printf("Argument 13: Number of columns in the test dataset (Number of input features + 1 (output variable)). The output variable should always be in the last column Ex: 5 \n\n");
    //     printf("Example:\n--------\n~$ ./MLP 3 4,5,5 softmax,relu,tanh 1 sigmoid 0.01 10000 data/data_train.csv 1096 5 data/data_test.csv 275 5\n\n");

    //     exit(0);
    // }


    // Create memory for training parameters struct
    param = (parameters*)malloc(sizeof(parameters));

    // Number of hidden layers
    param->n_hidden = atoi(new_argv[1]);
    // Sanity check of number of hidden layers
    if (param->n_hidden < 0) {
        printf("Error: Number of hidden layers should be >= 0\n");
        exit(0);
    }

    // Size of each hidden layer
    param->hidden_layers_size = (int*)malloc(param->n_hidden * sizeof(int));
    int i;
    char* tok;
    for (i = 0, tok = strtok(new_argv[2], ","); i < param->n_hidden; i++) {
        param->hidden_layers_size[i] = atoi(tok);
        // Sanity check of size of hidden layer
        if (param->hidden_layers_size[i] <= 0) {
            printf("Error: Hidden layer sizes should be positive\n");
            exit(0);
        }
        tok = strtok(NULL, ",");
    }

    // Hidden activation functions - Activation functions for each hidden layer
    param->hidden_activation_functions = (int*)malloc(param->n_hidden * sizeof(int));
    for (i = 0, tok = strtok(new_argv[3], ","); i < param->n_hidden; i++) {
        if (strcmp(tok, "identity") == 0) {
            param->hidden_activation_functions[i] = 1;
        }
        else if (strcmp(tok, "sigmoid") == 0) {
            param->hidden_activation_functions[i] = 2;
        }
        else if (strcmp(tok, "tanh") == 0) {
            param->hidden_activation_functions[i] = 3;
        }
        else if (strcmp(tok, "relu") == 0) {
            param->hidden_activation_functions[i] = 4;
        }
        else if (strcmp(tok, "softmax") == 0) {
            param->hidden_activation_functions[i] = 5;
        }
        else {
            printf("Error: Invalid value for hidden activation function\n");
            printf("Input either identity or sigmoid or tanh or relu or softmax for hidden activation function\n");
            exit(0);
        }

        tok = strtok(NULL, ",");
    }

    // Output layer size
    param->output_layer_size = atoi(new_argv[4]);
    if (param->output_layer_size <= 0) {
        printf("Output layer size should be positive\n");
        exit(0);
    }

    // Output activation function
    if (strcmp(new_argv[5], "identity") == 0) {
        param->output_activation_function = 1;
    }
    else if (strcmp(new_argv[5], "sigmoid") == 0) {
        param->output_activation_function = 2;
    }
    else if (strcmp(new_argv[5], "tanh") == 0) {
        param->output_activation_function = 3;
    }
    else if (strcmp(new_argv[5], "relu") == 0) {
        param->output_activation_function = 4;
    }
    else if (strcmp(new_argv[5], "softmax") == 0) {
        param->output_activation_function = 5;
    }
    else {
        printf("Error: Invalid value for output activation function\n");
        printf("Input either identity or sigmoid or tanh or relu or softmax for output activation function\n");
        exit(0);
    }


    // L2 Regularization parameter
    param->learning_rate = atoi(new_argv[6]);

    // Max. number of iterations
    param->n_iterations_max = atoi(new_argv[7]);
    if (param->n_iterations_max <= 0) {
        printf("Max. number of iterations value should be positive\n");
        exit(0);
    }

    // Momentum
    //param->momentum = atoi(new_argv[6]);

    // Get the parameters of the train dataset
    //char* train_filename = new_argv[8];
    //param->train_sample_size = atoi(new_argv[9]);
    // Feature size = Number of input features + 1 output feature
    //param->feature_size = atoi(new_argv[10]);

    // Create 2D array memory for the dataset
    //param->data_train = (double**)malloc(param->train_sample_size * sizeof(double*));
    //for (i = 0; i < param->train_sample_size; i++)
    //    param->data_train[i] = (double*)malloc(param->feature_size * sizeof(double));

    // Read the train dataset from the csv into the 2D array
    //read_csv(train_filename, param->train_sample_size, param->feature_size, param->data_train);
    
    double test_lines[][5] = {
        {1.602, 6.1251, 0.5292399999999999, 0.4788600000000001, 0},
        {-2.2918, -7.2570000000000014, 7.9597, 0.9211, 1},
        {-0.6907800000000001, -0.5007699999999999, -0.35417, 0.47498, 1},
        {1.6408, 4.2503, -4.9023, -2.6621, 1},
        {3.577, 2.4004, 1.8908, 0.73231, 0},
    };

    int test_sample_size = sizeof(test_lines) / sizeof(test_lines[0]);
    int feature_size = sizeof(test_lines[0]) / sizeof(double);

    // Get the parameters of the test dataset
    //char* test_filename = new_argv[11];
    //param->test_sample_size = atoi(new_argv[12]);
    // Feature size = Number of input features + 1 output feature
    //param->feature_size = atoi(new_argv[13]);

   // Create 2D array memory for the test data
    param->data_test = (double **)malloc(test_sample_size * sizeof(double *));
    for (int i = 0; i < test_sample_size; i++) {
        param->data_test[i] = (double *)malloc(feature_size * sizeof(double));
        memcpy(param->data_test[i], test_lines[i], feature_size * sizeof(double));
    }

    param->test_sample_size = test_sample_size;
    param->feature_size = feature_size;

    // Read the test dataset from the csv into the 2D array
    //read_csv(test_filename, param->test_sample_size, param->feature_size, param->data_test);

    // Total number of layers
    int n_layers = param->n_hidden + 2;

    // Save the sizes of layers in an array
    layer_sizes = (int*)calloc(n_layers, sizeof(int));

    layer_sizes[0] = param->feature_size - 1;
    layer_sizes[n_layers-1] = param->output_layer_size;

    for (i = 1; i < n_layers-1 ; i++)
        layer_sizes[i] = param->hidden_layers_size[i-1];

    // Create memory for the weight matrices between layers
    // weight is a pointer to the array of 2D arrays between the layers
    param->weight = (double***)calloc(n_layers - 1, sizeof(double**));

    // Each 2D array between two layers i and i+1 is of size ((layer_size[i]+1) x layer_size[i+1])
    // The weight matrix includes weights for the bias terms too
    for (i = 0; i < n_layers-1; i++)
        param->weight[i] = (double**)calloc(layer_sizes[i]+1, sizeof(double*));

    int j;
    for (i = 0; i < n_layers-1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            param->weight[i][j] = (double*)calloc(layer_sizes[i+1], sizeof(double));

    double weights[] = {
    0.725865, 0.441536, -0.799100, 0.009719, 0.445643, -0.595062, -0.250179, 0.208894, 0.276722, 0.190040, -0.046664, 0.763025, -0.214591, -0.399624, -0.743524, 0.735057, 0.204196, -0.515306, 0.641723, -0.267668,
    0.011293, 0.240472, 0.452365, 0.149054, -0.471252, 0.584530, -0.208878, -0.344829, 0.160482, 0.039268, 0.686929, 0.069851, -0.335692, 0.704326, -0.736927, -0.706546, -0.707233, -0.170609, 0.318845, 0.385986,
    -0.797066, -0.544316, 0.332514, -0.195160, -0.127443, 0.405487, -0.276599, -0.739743, 0.706677, -0.428210, -0.181118, -0.093471, 0.574518, -0.526563, -0.726662, -0.647147, -0.746626, -0.150224, -0.199683, 0.180217,
    0.661625, -0.322602, -0.528113, -0.431437, -0.429017, -0.452627, -0.327129, -0.325360, 0.160116, 0.749951, -0.733778, 0.178550, -0.541029, 0.356270, 0.768002, 0.112665, -0.033648, -0.269000, 0.185479, -0.177941,
    0.099907, -0.994370, 0.701389, -0.158393, -0.674160
    };

    int weightIndex = 0;
    for (int i = 0; i < n_layers - 1; i++) {
        for (int j = 0; j < layer_sizes[i] + 1; j++) {
            for (int k = 0; k < layer_sizes[i + 1]; k++) {
                param->weight[i][j][k] = weights[weightIndex++];
            }
        }
    }

    // Train the neural network on the train data
    // printf("Training:\n");
    // printf("---------\n");
    // mlp_trainer(param, layer_sizes);
    // printf("\nDone.\n\n");

    //save the trained weights to a file
    // FILE *fp = fopen("weights.txt", "w");
    // for (i = 0; i < n_layers-1; i++) {
    //     for (j = 0; j < layer_sizes[i]+1; j++) {
    //         for (int k = 0; k < layer_sizes[i+1]; k++) {
    //             fprintf(fp, "%lf ", param->weight[i][j][k]);
    //         }
    //         fprintf(fp, "\n");
    //     }
    //     fprintf(fp, "\n");
    // }
    // fclose(fp);

    // Classify the test data using the trained parameter weights
    //printf("Classifying:\n");
    //printf("------------\n");
    uint8_t accuracy = mlp_classifier(param, layer_sizes);
    //while(1)
        //simpleserial_get();
    //printf("\nDone.\nOutput file generated\n");

    // Free the memory allocated in Heap
    for (i = 0; i < n_layers-1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            free(param->weight[i][j]);

    for (i = 0; i < n_layers-1; i++)
        free(param->weight[i]);

    free(param->weight);

    free(layer_sizes);

    for (i = 0; i < param->train_sample_size; i++)
        free(param->data_train[i]);

    for (i = 0; i < param->test_sample_size; i++)
        free(param->data_test[i]);

    free(param->data_train);
    free(param->data_test);
    free(param->hidden_activation_functions);
    free(param->hidden_layers_size);
    free(param);
    
    
    simpleserial_put('r', 1, (uint8_t*)&accuracy);
    
    return 0x00;
}

int main(void) {
    // Initialize UART for serial communication
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();

    // Add a command to the SimpleSerial module
    simpleserial_addcmd('a', 0, mlp);
    //put some value so we can verify if we cna read them.

    for (;;)
    {
        simpleserial_get();
    }

    return 0;
}
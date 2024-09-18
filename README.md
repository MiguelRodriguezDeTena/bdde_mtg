Este repositorio incluye lo necesario para poner en funcionamiento el proyecto en Azure Databricks. Se necesita:

- Recurso de Databricks  
- Clúster de cómputo personal  
- Cuenta de almacenamiento ADLS GEN 2, en concreto Blob Storage.

Se añade el código, así como notebooks que son utilizados para cada uno de los flujos de trabajo.

Para crear los flujos de trabajo, la estructura será la siguiente:

**Predict \-**  
   
Bronze\_pipeline \-\> Silver\_pipeline \-\> gold\_pipeline \-\> Predict\_pipeline

**Train** 

Bronze\_pipeline \-\> Silver\_pipeline \-\> gold\_pipeline \-\> Train\_pipeline

Este notebook funciona también con parámetros que deben ser configurados a nivel de flujo de trabajo:

    "parameters": [  
      {  
        "name": "account",  
        "default":   
      },  
      {  
        "name": "account_key",  
        "default": "null"  \#poner aquí la clave de cuenta  
      },  
      {  
        "name": "config_dir",  
        "default": "/dbfs/FileStore/config.yaml" \#este archivo configurable también está provisto en el repositorio. Cambiar el email de usuario por el vuestro. Recomiendo usar esta ubicación para que lo guardéis y lo lea el programa.  
      },  
      {  
        "name": "mode",  
        "default": "predict"  #aquí se marca predict o train dependiendo de qué trabajo es.  
      },  
      {  
        "name": "root_dir",  
        "default": "abfss://mtg@{account}.dfs.core.windows.net"  
         #cambiar la cuenta de usuario por la vuestra. Crear el contenedor “mtg” en vuestra cuenta ADLS GEN 2 o crear con el nombre que guste siempre y cuando lo cambiéis en esta configuración  
      }  
    ]  

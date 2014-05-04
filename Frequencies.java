import java.io.*;
import java.util.*;

public class Frequencies{
    public static void main(String[] args) throws IOException{


        // Open file for reading
        List<String> data = new ArrayList<String>();
        for(int i = 0; i < 1000000; i++)
        {
            data.add("Alice");
            data.add("Bob");
            data.add("Charlie");
            data.add("Dan");
            data.add("Edith");
            data.add("Frank");
        }

        Map<String, Integer> result = new HashMap<String, Integer>();

        // Start timer
        final long startTime = System.nanoTime();  
        
        for(String item: data){
            if (!result.containsKey(item))
                result.put(item, 1);
            else
                result.put(item, result.get(item) + 1);
        }
        // End timer
        final long endTime = System.nanoTime();

        System.out.printf("Time elapsed: %f ms\n", 
                          (float)(endTime - startTime) / 1e6);
    }
}



package graphsearch.alpha;

import static java.lang.Math.abs;
import java.util.ArrayList;
import java.util.Iterator;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Ars Magna
 */
public class PII {
    // hacer static final?????
    private final int numDocs       = 1;// eliminar 
    private final int numQueries    = 2;
    private final int numSugg       = 4;
    private final int epsilon       = 10;


public ArrayList<Term> getFinalDoc(ArrayList<Term> DocOriginal) {
    
    ArrayList<Term> DocFinal = new ArrayList<Term>();
    Iterator<Term> itrDocOriginal = DocOriginal.iterator();
    
    for(int k = 0; (k < numQueries && itrDocOriginal.hasNext()); k++){ // funciona esta cosa??

        Term term = new Term();
        ArrayList<Long> posFinal = new ArrayList<Long>();
        
        Term Aux = itrDocOriginal.next();
        term.setTerm(Aux.getTerm());
        Iterator<Long> itrPos = Aux.getPositions().iterator();
        
        while (itrPos.hasNext()) {
            Long pos = itrPos.next();
            if(isOK_QueryTerm(pos,k,DocOriginal)){// ver .equals?
                posFinal.add(pos);
            }
        }
        term.setPositions(posFinal);

        DocFinal.add(term);
    }

    for(int k = numQueries; k < DocOriginal.size(); k++){ // revisar limites de integracion

	Term term = new Term();
        ArrayList<Long> posFinal = new ArrayList<Long>();
        
        Term Aux = itrDocOriginal.next();
        term.setTerm(Aux.getTerm());
        Iterator<Long> itrPos = Aux.getPositions().iterator();
        
        while (itrPos.hasNext()) {//ver si no tienes coincidencias coloco el string igual o no?
            Long pos = itrPos.next();
            if(isOK_RegularTerm(pos,DocFinal)){// pasar como argumento pos
                posFinal.add(pos);
            }
        }
        term.setPositions(posFinal);

        DocFinal.add(term);
     
    }	
        
        return DocFinal;
}
     



     private boolean isOK_QueryTerm(Long x,int exception, ArrayList<Term> Doc){
     	
        if(numQueries<=1){
            return true;
        }else{
            for (int i = 0; i < numQueries; i++) {
                if(i!=exception){ // no se puede validar a si misma
                    Iterator<Long> itrPos = Doc.get(i).getPositions().iterator();
                    while (itrPos.hasNext()) {                        
                        if (abs(itrPos.next()-x) <= epsilon) {
                            return true;
                        }
                    }
                }
            }

        }
        
         return false;
     }
    
     
     private boolean isOK_RegularTerm(Long x, ArrayList<Term> Doc){
     
        for (int i = 0; i < numQueries; i++) {// estoy seguro que parte desde > numqueries?
            Iterator<Long> itrPos = Doc.get(i).getPositions().iterator();
                while (itrPos.hasNext()) {                        
                    if (abs(itrPos.next()-x) <= epsilon) {
                        return true;
                    }
                }
                
        }
         return false;
     }
     
     
     
     
     
     
     
     
     
     
     
     
    public int getNumDocs() {
        return numDocs;
    }

    public int getNumQueries() {
        return numQueries;
    }

    public int getNumSugg() {
        return numSugg;
    }

    public int getEpsilon() {
        return epsilon;
    }
     
     
   
}

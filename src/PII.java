import java.util.ArrayList;
import java.util.Iterator;

import static java.lang.Math.abs;

/**
 * @author David Torres M.
 */


/**
 * II : Inverted Index
 * PII: Pseudo Inverted Index
 */
public class PII {
    // hacer static final?????
    private final int numQueries    = 1;
    private final int epsilon       = 10;


    public ArrayList<Term> getPII(ArrayList<Term> II) {

        ArrayList<Term> PII = new ArrayList<Term>();
        ArrayList<Integer> nonEmpty = nonEmptyIndexes(II);

        for (int i = 0; i < numQueries ; i++) {
            Term aux = new Term();
            aux.setTerm(II.get(i).getTerm());
            PII.add(aux);
        }

        switch (nonEmpty.size()) {
            case 0:  System.out.println("Documento sin coincidencias con los Query Terms");
                return PII;
            case 1:  PII.get(nonEmpty.get(0)).setPositions(II.get(nonEmpty.get(0)).getPositions());
                break;
            default: //rellenar con los terms que quedan no vacios luego de "operar" entre ellos
                for (int i:nonEmpty){
                    ArrayList<Integer> aux = new ArrayList<>(); int x;
                    Iterator<Integer> itrPos = II.get(i).getPositions().iterator();
                    while (itrPos.hasNext()) {
                        x = itrPos.next();
                        if (isOK_QueryTerm(x,i,II,nonEmpty))
                            aux.add(x);
                    }
                    PII.get(i).setPositions(aux);
                }
                break;
        }


        // Desde aca se comienzan a filtar las palabras regulares-normales del documento

        nonEmpty = nonEmptyIndexes(PII);
        int maxFor = II.size();
        for (int i = numQueries; i < maxFor; i++) {
            ArrayList<Integer> aux = new ArrayList<>(); int x;
            Iterator<Integer> itrPos = II.get(i).getPositions().iterator();
            while (itrPos.hasNext()) {
                x = itrPos.next();
                if (isOK_RegularTerm(x,PII,nonEmpty))
                    aux.add(x);
            }

            if(!aux.isEmpty()){// solo si tiene elementos entonces agrego a la lista de PII
                Term aux2 = new Term();
                aux2.setTerm(II.get(i).getTerm());
                aux2.setPositions(aux);
                PII.add(aux2);
            }
        }


        return PII;
    }




    private boolean isOK_QueryTerm(int x, int exception, ArrayList<Term> II, ArrayList<Integer> nonEmpty){
        for (Integer i : nonEmpty) { // no se puede validar con arreglos vacios
            if (i != exception) { // no se puede validar a si misma
                Iterator<Integer> itrPos = II.get(i).getPositions().iterator();
                while (itrPos.hasNext()) {
                    if (abs(itrPos.next() - x) <= epsilon) {
                        return true;
                    }
                }
            }
        }

        return false;
    }


    private boolean isOK_RegularTerm(int x, ArrayList<Term> PII, ArrayList<Integer> nonEmpty){
        for (int i:nonEmpty) {
            Iterator<Integer> itrPos = PII.get(i).getPositions().iterator();
            while (itrPos.hasNext()) {
                if (abs(itrPos.next() - x) <= epsilon) {
                    return true;
                }
            }
        }
        return false;
    }



    private ArrayList<Integer> nonEmptyIndexes(ArrayList<Term> II){
        ArrayList<Integer> nonEmpty = new ArrayList<>();
        for (int i = 0; i < numQueries ; i++) {
            if (!II.get(i).getPositions().isEmpty()) {
                nonEmpty.add(i);
            }
        }
        return nonEmpty;
    }





    public int getNumQueries() {
        return numQueries;
    }

    public int getEpsilon() {
        return epsilon;
    }
     
     
   
}

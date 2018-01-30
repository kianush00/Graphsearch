
import java.util.ArrayList;
import java.util.Iterator;
/**
 *
 * @author David Torres
 */
public class Term {
    private String term;
    private ArrayList<Integer> positions = new ArrayList<Integer>();


    public void printTerm(){
        System.out.print(this.term);
        Iterator<Integer> allPositions = positions.iterator();
            while(allPositions.hasNext()){
                System.out.print(" "+allPositions.next());
            }
        System.out.print("\n");    
    }
    

    public String getTerm() {
        return term;
    }

    public void setTerm(String term) {
        this.term = term;
    }

    public ArrayList<Integer> getPositions() {
        return positions;
    }

    public void setPositions(ArrayList<Integer> positions) {
        this.positions = positions;
    }






// podria utilizarse algun metodo de comparacion
/*    public boolean isEqual(Term term2){
        boolean resp = true;
        if(term2.getTerm().equals(term) && positions.size()==term2.getPositions().size()){
            Iterator<Integer> Pos1 = positions.iterator();
            Iterator<Integer> Pos2 = term2.getPositions().iterator();

            while(Pos1.hasNext()){
                if(Pos1.next().equals(Pos2.next()) == false){ // OJO .equals es menos optimo que compare, parece
                    resp = false;
                    break;
                }
            }
        }else{
            resp = false;
        }
        return resp;
    }


    public boolean isEqual2(Term term2){ // ES MAS LENTO??????
        if(term2.getTerm().equals(term) && positions.equals(term2.getPositions())){
            return true;
        }else{
            return false;
        }
    }*/




}

package graphsearch.alpha;

import java.util.ArrayList;
import java.util.Iterator;

public class Term {
    private String term;
    private ArrayList<Long> positions = new ArrayList<Long>();

    public String getTerm() {
        return term;
    }

    public void setTerm(String term) {
        this.term = term;
    }

    public ArrayList<Long> getPositions() {
        return positions;
    }

    public void setPositions(ArrayList<Long> positions) {
        this.positions = positions;
    }

    public void printTerm(){
        System.out.print(this.term);
        Iterator<Long> allPositions = positions.iterator();
            while(allPositions.hasNext()){
                System.out.print(" "+allPositions.next());
            }
        System.out.print("\n");    
    }
    
    public boolean isEqual(Term term2){
        boolean resp = true;

        if(term2.getTerm().equals(term) && positions.size()==term2.getPositions().size()){
            Iterator<Long> Pos1 = positions.iterator();
            Iterator<Long> Pos2 = term2.getPositions().iterator();

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
    }
}

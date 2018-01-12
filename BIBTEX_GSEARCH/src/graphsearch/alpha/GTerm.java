package graphsearch.alpha;

import java.util.Stack;

public class GTerm {
	private String term;
	private Stack<Long> positions = new Stack<>();
	private int frequency = 0;

	public GTerm(String term) {
		this.term = term;
	}
	
	public void addPosition(long pos) {
		this.positions.push(pos);
	}
	
	public void addPositions(Stack<Long> positions) {
		this.positions = positions;
	}
	
	public String getPositionsString() {
		return this.positions.toString();
	}
	
	public String getTerm() {
		return this.term;
	}
	
	public Stack<Long> getPositions() {
		return this.positions;
	}
	
	/**
	 * Calculate a new GTerm containing only the intersection
	 * positions of GTerm with a query term */
	public GTerm getIntersection(GTerm t2, int epsilon) {
		
		Long d_f = this.positions.firstElement();
		Long d_l = this.positions.lastElement();
		Long q_f = t2.getPositions().firstElement();
		Long q_l = t2.getPositions().lastElement();
		
		GTerm filteredTerm = new GTerm(this.term);
		
		// Do the arrays comparison only if we have some intersection
		if((q_l+epsilon >= d_f) || (d_l+epsilon >= q_f)) {
			for(Long pD:this.positions) {
				for(Long pQ:t2.getPositions()) {
					if(Math.abs(pD-pQ) <= epsilon)
						filteredTerm.addPosition(pD);
				}
			}
		}
		
		return filteredTerm;
	}
	
	public void print() {
		System.out.println(this.term + "  " + this.getPositionsString());
	}
}

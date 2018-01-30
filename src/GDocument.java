

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;

public class GDocument {
	int docId;
	Stack<GTerm> gTermStack = new Stack<>();
	
	final static Charset ENCODING = StandardCharsets.UTF_8;
	
	public GDocument(int docId) {
		this.docId = docId;
	}
	
	public void addTerm(String term) {
		this.gTermStack.push(new GTerm(term));
	}
	
	public void addTerm(GTerm gTerm) {
		this.gTermStack.push(gTerm);
	}
	
	public void addPositions(long pos) {
		this.gTermStack.peek().addPosition(pos);
	}
	
	/** NO ESTÁ LISTO  ....*/
/*	public GDocument applyQueryFilter(String[] queryStrings) {
		
		String[] searchStringsStemmed = this.kstemQueryStrings(queryStrings);
		
		Stack<GTerm> queryGTerms = new Stack<>();
		Stack<GTerm> noQueryGTerms = new Stack<>();
		
		for(String qString : queryStrings)
			queryGTerms.push(new GTerm(qString));
		
		for(GTerm gt : this.gTermStack) {		
			if(Arrays.asList(searchStringsStemmed).contains(gt.getTerm())){
				for(GTerm q:queryGTerms) {
					if(q.getTerm().contentEquals(gt.getTerm()))
						q.addPositions(gt.getPositions());					
				}
				
			} else {
				noQueryGTerms.push(gt);
			}
		}
		
		GDocument gDoc = new GDocument(this.docId);
		for(GTerm gT1:queryGTerms) {
			for(GTerm gT:queryGTerms) {
				
			}
			gDoc.addTerm(gT1);
		}
		for(GTerm gT2:noQueryGTerms) {
			gDoc.addTerm(gT2);
		}
		
		return gDoc;
	}*/
	
	
	/**
	 * Get an array of GTerm with the format needed 
	 * by David's algorithm.
	 * The first terms are query_terms (queryGTerms) followed 
	 * by no_query_terms (noQueryGTerms)
	 * */
	private GTerm[] getFormattedGTerms(String[] queryStrings) {
		Stack<GTerm> queryGTerms = new Stack<>();
		Stack<GTerm> noQueryGTerms = new Stack<>();
		String[] searchStringsStemmed = this.kstemQueryStrings(queryStrings);
		
		for(GTerm gt : this.gTermStack) {
			if(Arrays.asList(searchStringsStemmed).contains(gt.getTerm())){
				queryGTerms.push(gt);
			} else {
				noQueryGTerms.push(gt);
			}
		}

		GTerm[] output = new GTerm[queryGTerms.size() + noQueryGTerms.size()];
		int counter=0;
		while(!queryGTerms.isEmpty()) {
			output[counter++] = queryGTerms.pop();
		}
		while(!noQueryGTerms.isEmpty()) {
			output[counter++] = noQueryGTerms.pop();
		}
		return output;
	}
	
	
	
	/**
	 * This method get the content of the file 
	 * needed by David's algorithms 
	 * */
	private List<String> getFormattedOutputContent(String[] queryStrings) {
		
		/** This method get the formatted document with the query terms 
		 * in the first rows and then the other terms (no query terms) */
		GTerm[] formattedGTerms = this.getFormattedGTerms(queryStrings);
		
		List<String> output = new ArrayList<String>();
		for (int i=0; i<formattedGTerms.length; i++) {
			output.add(formattedGTerms[i].getTerm() + " " + formattedGTerms[i].getPositionsString());
		}
		return output;
	}
	
	
	/** 
	 * This is a test method.
	 * Should be later modified or removed
	 * @remove later
	 * */
	public ArrayList<Term> getDavidFormattedOutputContent(String[] queryStrings) {
		// This method get the formatted document with the query terms in the first rows and then
		// the other terms (no query terms)
		GTerm[] formattedGTerms = this.getFormattedGTerms(queryStrings);
		ArrayList<Term> output = new ArrayList<Term>();


		for (int i=0; i<formattedGTerms.length; i++) {
			Term term = new Term();
			ArrayList<Integer> positions = new ArrayList<>();
			
			term.setTerm(formattedGTerms[i].getTerm());

			for (Long pos:formattedGTerms[i].getPositions()){
				try{
					int val = Math.toIntExact(pos);
					positions.add(val);
				}catch(ArithmeticException e){
					System.out.println("overflows Long->Integer");
				}
			}

			term.setPositions(positions);
			output.add(term);
		}
		
		return output;
	}
	
	
	/**
	 * Create a text document with content of the GDocument
	 * @throws  
	 * */
	public void saveDocument(String aFileName, String[] queryStrings) {

		List<String> aLines = getFormattedOutputContent(queryStrings);
		
		try {
			writeSmallTextFile(aLines, aFileName);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/** 
	 * This is a test method.
	 * Should be later modified or removed
	 * @remove later
	 * */	
	public void saveDocumentFiltered(String[] queryStrings) {
		
		ArrayList<Term> terms = getDavidFormattedOutputContent(queryStrings);
		//
		PII filter = new PII();
		ArrayList<Term> filteredDoc = filter.getPII(terms);
		Iterator<Term> itDocuments = filteredDoc.iterator();
		
		while(itDocuments.hasNext()) {
			itDocuments.next().printTerm();
		}
	}
	
	
	private void writeSmallTextFile(List<String> aLines, String aFileName) throws IOException {
		    Path path = Paths.get(aFileName);
		    Files.write(path, aLines, ENCODING);
		  }
	
	/**
	 * Apply KStem to the query string
	 * */
	private String[] kstemQueryStrings(String[] queryStrings) {
		System.out.println("Parsing the search string using Krovetz-Stemmer...");
		for(int i=0; i<queryStrings.length; i++) {
			String temp = LuceneUtils.stemKrovetz(queryStrings[i]);
			System.out.println(queryStrings[i]+" -> "+temp);
			queryStrings[i] = temp;
		}
		return queryStrings;
	}
	
	/**
	 * Concatenate two GTerm arrays
	 * */
	public GTerm[] concat(GTerm[] a, GTerm[] b) {
		   int aLen = a.length;
		   int bLen = b.length;
		   GTerm[] c= new GTerm[aLen+bLen];
		   System.arraycopy(a, 0, c, 0, aLen);
		   System.arraycopy(b, 0, c, aLen, bLen);
		   return c;
		}
	
}


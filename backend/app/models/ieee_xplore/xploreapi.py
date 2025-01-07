import math
import urllib
import xml.etree.ElementTree as ET
import json

class XPLORE:
 
    # API endpoint
    endpoint = "http://ieeexploreapi.ieee.org/api/v1/search/articles"
    
    def __init__(self, api_key):

    	# API key
        self.api_key = api_key

    	# flag that some search criteria has been provided
        self.query_provided = False

        # flag that article number has been provided, which overrides all other search criteria
        self.using_article_number = False

        # flag that a boolean method is in use
        self.using_boolean = False

        # flag that a facet is in use
        self.using_facet = False

        # flag that a facet has been applied, in the event that multiple facets are passed
        self.facet_applied = False 

        # data type for results; default is json (other option is xml)
        self.output_type = 'json'

        # data format for results; default is raw (returned string); other option is object
        self.output_data_format = 'raw'

        # default of 25 results returned
        self.result_set_max = 25

        # maximum of 200 results returned
        self.result_set_max_cap = 50

        # records returned default to position 1 in result set
        self.start_record = 1

        # default sort order is ascending; could also be 'desc' for descending
        self.sort_order = 'desc'

        # field name that is being used for sorting
        self.sort_field = 'relevance'

        # array of permitted search fields for search_field() method
        self.allowed_search_fields = ['abstract', 'affiliation', 'article_number', 'article_title', 'author', 'boolean_text', 'content_type', 'd-au', 'd-pubtype', 'd-publisher', 'd-year', 'doi', 'end_year', 'facet', 'index_terms', 'isbn', 'issn', 'is_number', 'meta_data', 'open_access', 'publication_number', 'publication_title', 'publication_year', 'publisher', 'querytext', 'start_year', 'thesaurus_terms']

        # dictionary of all search parameters in use and their values
        self.parameters = {}

        # dictionary of all filters in use and their values
        self.filters = {}


    # ensuring == can be used reliably
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


    # ensuring != can be used reliably
    def __ne__(self, other):
        return not self.__eq__(other)


    # set the data type for the API output
    # string output_type   Format for the returned result (JSON, XML)
    # return void
    def data_type(self, output_type):

        output_type = output_type.strip().lower()
        self.output_type = output_type


    # set the data format for the API output
    # string output_data_format   Data structure for the returned result (raw string or object)
    # return void
    def data_format(self, output_data_format):

        output_data_format = output_data_format.strip().lower()
        self.output_data_format = output_data_format


    # set the start position in the
    # string start   Start position in the returned data
    # return void
    def starting_result(self, start):

        self.start_record = math.ceil(start) if (start > 0) else 1


    def maximum_results(self, maximum_results: int):
        """
        Sets the maximum number of results to be returned by the IEEE Xplore API.

        Parameters:
        maximum_results (int): The maximum number of results to return. If the value is less than or equal to 0, the default value of 25 is used. If the value exceeds the maximum allowed value (50), the maximum allowed value is used.

        Returns:
        None
        """
        self.result_set_max = math.ceil(maximum_results) if (maximum_results > 0) else 25
        if self.result_set_max > self.result_set_max_cap:
            self.result_set_max = self.result_set_max_cap



    # setting a filter on results
    # string filter_param   Field used for filtering
    # string value    Text to filter on
    # return void
    def results_filter(self, filter_param, value):

        filter_param = filter_param.strip().lower()
        value = value.strip()

        if len(value) > 0:
            self.filters[filter_param] = value
            self.query_provided = True

            # Standards do not have article titles, so switch to sorting by article number
            if (filter_param == 'content_type' and value == 'Standards'):
                self.results_sorting('publication_year', 'asc')


    # setting sort order for results
    # string field   Data field used for sorting
    # string order   Sort order for results (ascending or descending)
    # return void
    def results_sorting(self, field, order):

        field = field.strip().lower()
        order = order.strip()
        self.sort_field = field
        self.sort_order = order


    # shortcut method for assigning search parameters and values
    # string field   Field used for searching
    # string value   Text to query
    # return void
    def search_field(self, field, value):

        field = field.strip().lower()
        if field in self.allowed_search_fields:
            self.add_parameter(field, value)
        else:
            print("Searches against field " + field + " are not supported")


    # string value   Abstract text to query
    # return void
    def abstract_text(self, value):

        self.add_parameter('abstract', value)


    # string value   Affiliation text to query
    # return void
    def affiliation_text(self, value):

        self.add_parameter('affiliation', value)


    # string value   Article number to query
    # return void
    def article_number(self, value):

        self.add_parameter('article_number', value)


    # string value   Article title to query
    # return void
    def article_title(self, value):

        self.add_parameter('article_title', value)


    # string value   Author to query
    # return void
    def author_text(self, value):

        self.add_parameter('author', value)


    # string value   Author Facet text to query
    # return void
    def author_facet_text(self, value):

        self.add_parameter('d-au', value)


    # string value   Value(s) to use in the boolean query
    # return void
    def boolean_text(self, value):

        self.add_parameter('boolean_text', value)


    # string value   Content Type Facet text to query
    # return void
    def content_type_facet_text(self, value):

        self.add_parameter('d-pubtype', value)


    # string value   DOI (Digital Object Identifier) to query
    # return void
    def doi(self, value):

        self.add_parameter('doi', value)


    # string value   Facet text to query
    # return void
    def facet_text(self, value):

        self.add_parameter('facet', value)


    # string value   Author Keywords, IEEE Terms, and Mesh Terms to query
    # return void
    def index_terms(self, value):

        self.add_parameter('index_terms', value)


    # string value   ISBN (International Standard Book Number) to query
    # return void
    def isbn(self, value):

        self.add_parameter('isbn', value)


    # string value   ISSN (International Standard Serial number) to query
    # return void
    def issn(self, value):

        self.add_parameter('issn', value)


    # string value   Issue number to query
    # return void
    def issue_number(self, value):

        self.add_parameter('is_number', value)


    # string value   Text to query across metadata fields and the abstract
    # return void
    def meta_data_text(self, value):

        self.add_parameter('meta_data', value)


    # string value   Publication Facet text to query
    # return void
    def publication_facet_text(self, value):

        self.add_parameter('d-year', value)


    # string value   Publisher Facet text to query
    # return void
    def publisher_facet_text(self, value):

        self.add_parameter('d-publisher', value)


    # string value   Publication title to query
    # return void
    def publication_title(self, value):

        self.add_parameter('publication_title', value)


    # string or number value   Publication year to query
    # return void
    def publication_year(self, value):

        self.add_parameter('publication_year', value)


    def query_text(self, value: str) -> None:
        """
        Adds a query parameter for the 'querytext' field to the XPLORE API request.

        Parameters:
        value (str): The text to use in the query. This parameter is used to search across metadata fields, abstract, and document text.

        Returns:
        None

        This function calls the 'add_parameter' method with the 'querytext' parameter and the provided value. It adds the query parameter to the 'parameters' dictionary and sets the 'query_provided' flag to True.
        """
        self.add_parameter('querytext', value)



    # string value   Thesaurus terms (IEEE Terms) to query
    # return void
    def thesaurus_terms(self, value):

        self.add_parameter('thesaurus_terms', value)


    def add_parameter(self, parameter: str, value: str):
        """
        Adds a query parameter to the XPLORE API request.

        Parameters:
        parameter (str): The name of the data field to query.
        value (str): The text to use in the query.

        Returns:
        None

        This function adds a query parameter to the XPLORE API request. It first strips any leading or trailing whitespace from the value.
        If the value is not empty, it adds the parameter and value to the 'parameters' dictionary. It also sets flags based on the parameter
        to indicate the type of query being performed.
        """
        value = value.strip()

        if (len(value) > 0):

            self.parameters[parameter]= value

            # viable query criteria provided
            self.query_provided = True

            # set flags based on parameter
            if (parameter == 'article_number'):

                self.using_article_number = True

            if (parameter == 'boolean_text'):

                self.using_boolean = True

            if (parameter == 'facet' or parameter == 'd-au' or parameter == 'd-year' or parameter == 'd-pubtype' or parameter == 'd-publisher'):

                self.using_facet = True



    def call_api(self, debug_mode_off=True):
        """
        Calls the IEEE Xplore API with the specified query parameters.

        Parameters:
        debug_mode_off (bool): If True, the function will return the raw API response. If False, the function will process the response and return formatted data. Default is True.

        Returns:
        str or object: Raw API response string if debug_mode_off is False. Formatted data (XML or JSON object) if debug_mode_off is True and query criteria are provided. Prints an error message and returns None if no search criteria are provided.
        """
        string = self.build_query()

        if debug_mode_off is False:
            return string

        else:
            if self.query_provided is False:
                print ("No search criteria provided")

            data = self.query_api(string)
            formatted_data = self.format_data(data)
            return formatted_data



    # creates the URL for the API call
    # return string: full URL for querying the API
    def build_query(self):

        url = self.endpoint

        url += '?apikey=' + str(self.api_key)
        url += '&format=' + str(self.output_type)
        url += '&max_records=' + str(self.result_set_max)
        url += '&start_record=' + str(self.start_record)
        url += '&sort_order=' + str(self.sort_order)
        url += '&sort_field=' + str(self.sort_field)

        # add in search criteria
        # article number query takes priority over all others
        if (self.using_article_number):

            url += '&article_number=' + str(self.parameters['article_number'])

        # boolean query
        elif (self.using_boolean):

             url += '&querytext=(' + urllib.quote_plus(self.parameters['boolean_text']) + ')'

        else:

            for key in self.parameters:

                if (self.using_facet and self.facet_applied is False):

                    url += '&querytext=' + urllib.parse.quote(self.parameters[key]) + '&facet=' + key
                    self.facet_applied = True

                else:

                    url += '&' + key + '=' + urllib.parse.quote(self.parameters[key])


        # add in filters
        for key in self.filters:

            url += '&' + key + '=' + str(self.filters[key])
 
        return url


    # creates the URL for the API call
    # string url  Full URL to pass to API
    # return string: Results from API
    def query_api(self, url):
        import urllib.request
        content = urllib.request.urlopen(url).read()
        return content


    # formats the data returned by the API
    # string data    Result string from API
    def format_data(self, data):

        if self.output_data_format == 'raw':
            return data

        elif self.output_data_format == 'object':
            
            if self.output_type == 'xml':
                obj = ET.ElementTree(ET.fromstring(data))
                return obj

            else:
                obj = json.loads(data) 
                return obj

        else:
            return data

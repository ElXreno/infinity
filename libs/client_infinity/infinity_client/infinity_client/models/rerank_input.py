from typing import Any, Dict, Type, TypeVar, Tuple, Optional, BinaryIO, TextIO, TYPE_CHECKING

from typing import List


from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from typing import cast, List
from typing import Union
from typing import cast, Union
from ..types import UNSET, Unset






T = TypeVar("T", bound="RerankInput")


@_attrs_define
class RerankInput:
    """ Input for reranking

        Attributes:
            query (str):
            documents (List[str]):
            return_documents (Union[Unset, bool]):  Default: False.
            raw_scores (Union[Unset, bool]):  Default: False.
            model (Union[Unset, str]):  Default: 'default/not-specified'.
            top_n (Union[None, Unset, int]):
            max_query_tokens (Union[None, Unset, int]): Head-truncate the query to at most N tokens before scoring. Clamped
                to the model's server-side ceiling: a request may lower this but not raise it above the configured limit. Omit
                or null to use the server ceiling.
            max_tokens_per_doc (Union[None, Unset, int]): Head-truncate each document to at most N tokens before scoring
                (Cohere v2 compatible). Clamped to the model's server-side ceiling: a request may lower this but not raise it.
                Omit or null to use the server ceiling.
            max_pair_tokens (Union[None, Unset, int]): Cap the joined (query, document) pair to at most N tokens, trimming
                the longest side first. Clamped to the model's server-side ceiling: a request may lower this but not raise it.
                Omit or null to use the server ceiling.
     """

    query: str
    documents: List[str]
    return_documents: Union[Unset, bool] = False
    raw_scores: Union[Unset, bool] = False
    model: Union[Unset, str] = 'default/not-specified'
    top_n: Union[None, Unset, int] = UNSET
    max_query_tokens: Union[None, Unset, int] = UNSET
    max_tokens_per_doc: Union[None, Unset, int] = UNSET
    max_pair_tokens: Union[None, Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        query = self.query

        documents = self.documents



        return_documents = self.return_documents

        raw_scores = self.raw_scores

        model = self.model

        top_n: Union[None, Unset, int]
        if isinstance(self.top_n, Unset):
            top_n = UNSET
        else:
            top_n = self.top_n

        max_query_tokens: Union[None, Unset, int]
        if isinstance(self.max_query_tokens, Unset):
            max_query_tokens = UNSET
        else:
            max_query_tokens = self.max_query_tokens

        max_tokens_per_doc: Union[None, Unset, int]
        if isinstance(self.max_tokens_per_doc, Unset):
            max_tokens_per_doc = UNSET
        else:
            max_tokens_per_doc = self.max_tokens_per_doc

        max_pair_tokens: Union[None, Unset, int]
        if isinstance(self.max_pair_tokens, Unset):
            max_pair_tokens = UNSET
        else:
            max_pair_tokens = self.max_pair_tokens


        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "query": query,
            "documents": documents,
        })
        if return_documents is not UNSET:
            field_dict["return_documents"] = return_documents
        if raw_scores is not UNSET:
            field_dict["raw_scores"] = raw_scores
        if model is not UNSET:
            field_dict["model"] = model
        if top_n is not UNSET:
            field_dict["top_n"] = top_n
        if max_query_tokens is not UNSET:
            field_dict["max_query_tokens"] = max_query_tokens
        if max_tokens_per_doc is not UNSET:
            field_dict["max_tokens_per_doc"] = max_tokens_per_doc
        if max_pair_tokens is not UNSET:
            field_dict["max_pair_tokens"] = max_pair_tokens

        return field_dict



    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        query = d.pop("query")

        documents = cast(List[str], d.pop("documents"))


        return_documents = d.pop("return_documents", UNSET)

        raw_scores = d.pop("raw_scores", UNSET)

        model = d.pop("model", UNSET)

        def _parse_top_n(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        top_n = _parse_top_n(d.pop("top_n", UNSET))


        def _parse_max_query_tokens(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        max_query_tokens = _parse_max_query_tokens(d.pop("max_query_tokens", UNSET))


        def _parse_max_tokens_per_doc(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        max_tokens_per_doc = _parse_max_tokens_per_doc(d.pop("max_tokens_per_doc", UNSET))


        def _parse_max_pair_tokens(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        max_pair_tokens = _parse_max_pair_tokens(d.pop("max_pair_tokens", UNSET))


        rerank_input = cls(
            query=query,
            documents=documents,
            return_documents=return_documents,
            raw_scores=raw_scores,
            model=model,
            top_n=top_n,
            max_query_tokens=max_query_tokens,
            max_tokens_per_doc=max_tokens_per_doc,
            max_pair_tokens=max_pair_tokens,
        )


        rerank_input.additional_properties = d
        return rerank_input

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
